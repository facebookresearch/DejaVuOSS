import torchvision
from torchvision import transforms
from torchvision.models import resnet50, resnet101
import xmltodict
import os
from torchvision.datasets import ImageFolder

class ImageFolderIndexWithPath(ImageFolder): 
    """Takes a subset of ImageFolder and returns original
    ImageFolder index along with image/label
    for valid dataset pass offset: 1281167
    """
    def __init__(self, im_pth, xfrm, subset_idxs, offset=0): 
        super().__init__(
            im_pth, 
            transform = xfrm 
        )
        self.xfrm = xfrm
        self.subset_idxs = subset_idxs
        self.offset = offset
        
    def __len__(self): 
        return len(self.subset_idxs)
        
    def __getitem__(self, index: int):
        idx = self.subset_idxs[index] # get full imgnet index
        path, target = self.samples[idx - self.offset]
        sample = self.loader(path)
        sample = self.xfrm(sample)
        
        return sample, target, idx, path


class ImageFolderWithPath(ImageFolder): 
    """Takes a subset of ImageFolder and returns original
    ImageFolder index along with image/label
    for valid dataset pass offset: 1281167
    """
    def __init__(self, im_pth, xfrm): 
        super().__init__(
            im_pth, 
            transform = xfrm 
        )
        
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def __len__(self) -> int:
        return len(self.samples)


class ImageFolderIndex(ImageFolder): 
    """Takes a subset of ImageFolder and returns original
    ImageFolder index along with image/label
    for valid dataset pass offset: 1281167
    """
    def __init__(self, im_pth, xfrm, subset_idxs, offset=0): 
        super().__init__(
            im_pth, 
            transform = xfrm 
        )
        self.xfrm = xfrm
        self.subset_idxs = subset_idxs
        self.offset = offset
        
    def __len__(self): 
        return len(self.subset_idxs)
        
    def __getitem__(self, index: int):
        idx = self.subset_idxs[index] # get full imgnet index
        path, target = self.samples[idx - self.offset]
        sample = self.loader(path)
        sample = self.xfrm(sample)
        
        return sample, target, idx 

class InverseTransform:
    """inverses normalization of SSL transform """
    def __init__(self): 
        self.invTrans = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
        std = [ 1., 1., 1. ]),
        ])

    def __call__(self, x): 
        return self.invTrans(x)

class SSL_Transform: 
    """Transform applied to SSL examples at test time 
    """
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.ssl_xfrm = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                        ])
        
    def __call__(self, x): 
       return self.ssl_xfrm(x)

class crop_xfrm: 
    """Transform to get lower left crop of image 
    """
    def __init__(self, crop_frac = 0.3, use_normalize = True):
        self.crop_frac = crop_frac
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    
        if use_normalize: 
            self.xfrm = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                            ])
        else: 
            self.xfrm = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()
                            ])
        
    def __call__(self, x): 
        #get lower left corner
        horiz, vert = x.size
        height = int(self.crop_frac * vert)
        width = int(self.crop_frac * horiz)
        crop = transforms.functional.crop(x, top = 0, left = 0, 
                                          height = height, width = width)
        '''
        crop = transforms.functional.crop(x, top = vert - height, left = 0, 
                                          height = height, width = width)
        '''
        #transform for SSL 
        return self.xfrm(crop)

class AuxDataset(torchvision.datasets.ImageFolder):
    """pytorch dataset that returns a patch of an image outside of the bounding boxes
    along with the image index. The patch must have side lengths (in pixels) above a 
    threshold. If not, the dataset returns an uncropped version of the image along 
    with the good=-1 flag. 
    Inputs: 
        im_pth: path to imagenet train set 
        bbox_pth: path to imagenet train set bounding boxes 
        indices: subset of imagenet train set (all indices must have
            corresponding bounding boxes) 
        min_px: sidelength threshold in pixels (default 100) 
    Returns:
        given: crop transformed for SSL 
        good: whether there exists a large enough crop
        index: image index
    """
    def __init__(self, im_pth, bbox_pth, indices, min_px = 100, return_im_and_tgt = False, make_ffcv_dataset = False, log_fn=None): 
        """dataset size of full imagenet 
        """
        super().__init__(
            im_pth, 
            transform = None 
        )
        self.bbox_pth = bbox_pth 
        self.indices = indices 
        self.min_px = min_px 

        #flag to return the crop, original image, and label for RCDM reconstruction 
        #during RCDM reconstruction, we only use this loader to view crop/image, not for reconstruction itself
        self.return_im_and_tgt = return_im_and_tgt

        #flag to return crop, target, index, and 'good' parameter for bbox size validity -- only used for making bbox ffcv dataset used in online attacks 
        self.make_ffcv_dataset = make_ffcv_dataset

    
        #transform for squaring off and normalizing cropped piece of image 
        self.ssl_xfrm = transforms.Compose([transforms.ToPILImage(), SSL_Transform()]) 
        self.log_fn = log_fn        

    def mergeIntervals(self, intervals):
        """given a list of intervals, return a list with 
        intersecting intervals merged
        """
        intervals.sort()
        stack = []
        # insert first interval into stack
        stack.append(intervals[0])
        for i in intervals[1:]:
            # Check for overlapping interval,
            # if interval overlap
            if stack[-1][0] <= i[0] <= stack[-1][-1]:
                stack[-1][-1] = max(stack[-1][-1], i[-1])
            else:
                stack.append(i)

        return stack 

    def get_crop(self, im, bndboxes):
        """given tensor image and list of bounding boxes, 
           find the largest (by min side length) box outside the
           bounding boxes. If minimum side length is below the
           min_px threshold, returns None
           Inputs: 
               - im: torch tensor of image 
               - bndboxes: list of bounding box dicts
           Returns: 
               - the_box: largest box outside bounding boxes
                    if one exists above threshold, else None
        """

        #First get a list of where all x-values where bounding 
        #boxes start/end in order from left to right
        _, ymax, xmax = im.shape

        xlims = []
        for bbox in bndboxes: 
            xlims += [int(bbox['xmin']), int(bbox['xmax'])]
        xlims.sort()
        if xlims[0] > 0: 
            xlims.insert(0, 0)
        if xlims[-1] < xmax: 
            xlims.append(xmax)
            
        if xlims[-1] > xmax: 
            #ignore examples with bounding boxes 
            #outside image boundary
            return

        #for each x interval, find all y intervals outside 
        #bounding boxes. The combination of x and y interval creates
        #a candidate box 
        boxes = []
        #for each x interval between box edges
        for interval in zip(xlims[:-1], xlims[1:]): 
            # get list of bbox y limits for bboxes intersecting xinterval 
            y_intervals = [] 
            for bbox in bndboxes: 
                #if bbox intersects interval, add it to y_intervals
                intersects_interval = min(int(bbox['xmax']), interval[1]) - \
                                      max(int(bbox['xmin']), interval[0]) > 0
                if intersects_interval:
                    y_intervals.append([int(bbox['ymin']), int(bbox['ymax'])])

            #get merged y intervals -- potential boxes are in the negative of this 
            if y_intervals: 
                y_intervals = self.mergeIntervals(y_intervals)

            #get candidate boxes with the current x interval
            #from 0 to first edge 
            y_intervals.insert(0, [0,0])
            y_intervals.append([ymax, ymax])
            for inter_0, inter_1 in zip(y_intervals[0:], y_intervals[1:]): 
                box_xmin, box_xmax = interval[0], interval[1]
                box_ymin, box_ymax = inter_0[1], inter_1[0]
                if (box_ymax > box_ymin) and (box_xmax > box_xmin): 
                    bbox = {'xmin':box_xmin, 'xmax':box_xmax, 
                            'ymin':box_ymin, 'ymax':box_ymax,
                            'min len': min((box_ymax - box_ymin), (box_xmax - box_xmin))}
                    boxes.append(bbox)

        the_box = None
        if boxes: 
            #return the bounding box with the largest 
            #minimum side length, if it is larger than the threshold
            boxes.sort(key=lambda x: x['min len'])
            if boxes[-1]['min len'] > self.min_px: 
                the_box = boxes[-1]
        '''
        else:
            # print some parameters for debugging
            print('bndboxes: ', bndboxes)
            print('xlims: ', xlims)
            print('ymax: ', ymax, 'xmax: ', xmax)
        '''
        return the_box
    
    def __len__(self): 
        return len(self.indices)
        
    def __getitem__(self, index: int):
        #return cropped piece of image outside bbox and label
        #also return index in ImageLoader dataset
        index = self.indices[index]
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = transforms.ToTensor()(sample)
        
        #get bounding box data from file and parse into list  
        CLS = path.split('/')[-2]
        fname = path.split('/')[-1].split('.')[0] + '.xml'
        bbox_file = os.path.join(self.bbox_pth, 'Annotation', CLS, fname)
        if not os.path.exists(bbox_file):
            #print(f'No file path found: {bbox_file}')
            self.log_fn({'No file path found': bbox_file})
            return -1, -1, -1, index

        with open(bbox_file) as fd:
            bbox_info = xmltodict.parse(fd.read())

        if type(bbox_info['annotation']['object']) == list: 
            bndboxes = [b['bndbox'] for b in bbox_info['annotation']['object']]
        else: 
            bndboxes = [bbox_info['annotation']['object']['bndbox']]
            
        #Now finding largest cropping possible outside bounding box 
        #if no box exists, return original image with label -1
        crop_box = self.get_crop(sample, bndboxes)
        try:
            if crop_box: 
                cropped = sample[:, crop_box['ymin']:crop_box['ymax'], crop_box['xmin']:crop_box['xmax']] 
                given = self.ssl_xfrm(cropped)
                good = 1
            else: 
                cropped = sample
                given = self.ssl_xfrm(cropped)
                good = -1
                self.log_fn({'Cannot find correct bounding box': bbox_file})

            if self.return_im_and_tgt: 
                return given, good, sample, target
            elif self.make_ffcv_dataset: 
                given_PIL = transforms.ToPILImage()(cropped)
                return given_PIL, target, index, good
            else: 
                return given, good, index 
        except ValueError as verr:
            self.log_fn({'bbox_file ': bbox_file,
                         'value error': str(verr),
                         'coordinates': ' ymin: ' + str(crop_box['ymin']) + ' ymax: ' + str(crop_box['ymax']) + \
                         'xmin: ' + str(crop_box['xmin']) + ' xmax: ' + str(crop_box['xmax'])})
            #print('value error: ', verr)
            #print('ymin: ', crop_box['ymin'], 'ymax: ', crop_box['ymax'], \
            #      'xmin: ', crop_box['xmin'], 'xmax: ', crop_box['xmax'])
            return None, -1, -1, -1
