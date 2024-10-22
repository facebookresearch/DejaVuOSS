from groundingdino.util.inference import load_model, load_image, predict
import os
from torchvision.ops import box_convert
import torch
import torchvision
import json
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import argparse
from pathlib import Path
import dataclasses
import dicttoxml
from xml.dom.minidom import parseString
from torch.utils.data import DataLoader
from collections import Counter

'''
python grounding_dino_imagenet_annotator.py --annotation_save_dir <HOME>/Grounded-Segment-Anything/images/train/Annotation_Blurred \
								      --imagenet_dir <IMAGENET-HOME>/train  \
								      --idx_path <DEJAVU-OUTPUT>/imagenet_partition_out_debug/300_per_class/bbox_A_test.npy \
								      --config 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py' \
								      --grounded_checkpoint './groundingdino_swint_ogc.pth' \
								      --box_threshold 0.25  \
								      --text_threshold 0.2 \
								      --device cuda
'''

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

class Node:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name

@dataclasses.dataclass
class Bndbox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

@dataclasses.dataclass
class Object:
    name: str
    pose: str
    truncated: int
    difficult: int
    bndbox: Bndbox

@dataclasses.dataclass
class Size:
    width: int
    height: int
    depth: int

@dataclasses.dataclass
class Source:
    database: str

@dataclasses.dataclass
class Annotation:
    folder: str
    filename: str
    source: Source
    size: Size
    segmented: str
    object_wrapper: dict
    
''' Example xml structure
<annotation>
	<folder>n07809368</folder>
	<filename>n07809368_24395</filename>
	<source>
		<database>ImageNet database</database>
	</source>
	<size>
		<width>500</width>
		<height>393</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>n07809368</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>29</xmin>
			<ymin>41</ymin>
			<xmax>434</xmax>
			<ymax>366</ymax>
		</bndbox>
	</object>
</annotation>
'''

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses / dark glasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

class crop_xfrm: 
    """Transform to get lower left crop of image 
    """
    def __init__(self, use_normalize = True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if use_normalize: 
            self.xfrm = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        normalize,
                            ])
        else: 
            self.xfrm = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor()
                            ])
        
    def __call__(self, x): 
        #transform for SSL
        return self.xfrm(x)

def mergeIntervals(intervals):
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

def get_crop(im, bndboxes):
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
    print(ymax, xmax)
    bbox = bndboxes[0]

    x1 = int(bbox[0] * xmax) - int((bbox[2]*xmax)/2.)
    x2 = int(bbox[0] * xmax) + int((bbox[2]*xmax)/2.)
    y1 = int(bbox[1] * ymax) - int((bbox[3]*ymax)/2.)
    y2 = int(bbox[1] * ymax) + int((bbox[3]*ymax)/2.)
    print(x1,x2,y1,y2)
    for bbox in bndboxes: 
        print(bbox)
        xlims += [int(x1), int(x2)]
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
            intersects_interval = min(int(x2), interval[1]) - \
                                    max(int(x1), interval[0]) > 0
            if intersects_interval:
                y_intervals.append([int(y1), int(y2)])

        #get merged y intervals -- potential boxes are in the negative of this 
        if y_intervals: 
            y_intervals = mergeIntervals(y_intervals)

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

    #return box_xmax 
    the_box = None
    if boxes: 
        #return the bounding box with the largest 
        #minimum side length, if it is larger than the threshold
        boxes.sort(key=lambda x: x['min len'])
        if boxes[-1]['min len'] > 5: 
            the_box = boxes[-1]
            box_xmin, box_xmax = the_box['xmin'], the_box['xmax']
            box_ymin, box_ymax = max(the_box['ymin'], 100), the_box['ymax']
            cx = (box_xmin + int((box_xmax - box_xmin)/2.)) / xmax
            cy = (box_ymin + int((box_ymax - box_ymin)/2.)) / ymax
            w = (box_xmax - box_xmin) / xmax
            h = (box_ymax - box_ymin) / ymax
            return (cx,cy,w,h), the_box

    return the_box



def main(args):
    imagenet_dir = args.imagenet_dir
    annotation_save_dir = args.annotation_save_dir
    bbox_idx_path = args.idx_path
    config = args.config
    grounded_checkpoint = args.grounded_checkpoint
    box_threshold = args.box_threshold 
    text_threshold = args.text_threshold
    device = args.device

    tr_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        tr_normalize,
    ])
    imagenet_data = torchvision.datasets.ImageFolder(imagenet_dir, transform=transform)

    labels = []
    idxs = []

    # We get all the embeddings
    bbox_idx = np.load(bbox_idx_path)
    ds_set = ImageFolderIndex(imagenet_dir, SSL_Transform(), bbox_idx)
    ds_loader = DataLoader(ds_set, batch_size = 64, shuffle = False, num_workers=8) 

    for _, (x,y,idx) in enumerate(ds_loader): 
        x = x.cuda()
        with torch.no_grad():
            labels.append(y.numpy())
            idxs.append(idx.numpy())

    labels = np.concatenate(labels, axis = 0)
    idxs = np.concatenate(idxs)

    print('idxs len: ', len(idxs), 'labels len :', len(labels))

    model = load_model(config, grounded_checkpoint)
        
    annotations = []
    annotations_idxes = []
    stats = Counter()
    # Then we go through all of the examples in a class
    for idx, label in zip(idxs, labels):
        img_path = imagenet_data.samples[idx][0]
        image_source, image = load_image(img_path)

        print('img_path: ', img_path)

        image_path_split = img_path.split(os.sep)
        file_name = image_path_split[-1]
        folder_name = image_path_split[-2]

        folder_name_full_path = annotation_save_dir / folder_name
        #if os.path.exists(folder_name_full_path):
        #    continue

        print('image_source: ', image_source.shape)
        print('image: ', image.shape)

        height, width, depth = image_source.shape
    
        # We predict the bouding boxes of the specific object
        prompt = imagenet_classes[label]
        boxes, _, _ = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device="cuda",
        )
        if len(boxes) > 0:
            stats[prompt] += 1
            print('promopt: ', prompt)
            boxes = boxes * torch.Tensor([width, height, width, height])
            
            boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            
            boxes_obj = {}
            for box in boxes_xyxy:
                x0, y0, x1, y1 = box

                boxes_obj[Node('object')] = Object(folder_name, 'Unspecified', 0, 0, Bndbox(int(x0), int(y0), int(x1), int(y1)))

            print('idx boxes: ', boxes, 'label: ', label, 'idx: ', idx, 'image size: ', image.shape)

            annotation = Annotation(folder_name, file_name.split('.')[0],
                                    Source('ImageNet database'),
                                    Size(width, height, depth),
                                    0, 
                                    boxes_obj)
            annotation_dict = dataclasses.asdict(annotation)

            annotations.append(annotation)

            annotation_xml: str = dicttoxml.dicttoxml(annotation_dict,
                                        attr_type=False,
                                        custom_root='annotation',
                                        xml_declaration=False).decode().replace("<item>", "").replace("</item>", "").replace("<object_wrapper>", "").replace("</object_wrapper>", "")
            dom = parseString(annotation_xml)

            if not os.path.exists(folder_name_full_path):
                os.makedirs(folder_name_full_path)
            file_name_xml = file_name.split('.')[0] + '.xml'
            print('file_name_xml: ', file_name_xml)

            # write into xml file
            print('xml file full path: ', folder_name_full_path / file_name_xml)
            with open(folder_name_full_path / file_name_xml, "wb") as fl:
                dom_xml = dom.childNodes[0].toprettyxml(encoding='utf-8')
                fl.write(dom_xml)

            annotations_idxes.append(idx)

    print('idxs len: ', len(idxs), 'labels len :', len(labels))
    print('labels: ', len(labels))
    print('annotations: ', len(annotations))

    with open(annotation_save_dir / 'summary.json', 'w') as fl:
        json.dump(stats, fl)

    np.save(annotation_save_dir / 'bbox_annotation_idx', annotations_idxes)

def parse_args():
    parser = argparse.ArgumentParser("Arguments for Grounding Dino based Annotator")

    parser.add_argument("--annotation_save_dir", type=Path, required=True)
    parser.add_argument("--imagenet_dir", type=Path, required=True)
    parser.add_argument("--idx_path", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--grounded_checkpoint", type=Path, required=True)
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)