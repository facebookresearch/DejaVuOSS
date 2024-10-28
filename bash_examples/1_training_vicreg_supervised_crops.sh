# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python train_SSL_crops.py \
		--config-file configs/supervised_cfg.yaml \
		--dist.use_submitit 1 \
		--data.train_dataset $BETONS_FOLDER/300npc_modelA_crops_debug.beton \
		--data.val_dataset $BETONS_FOLDER/300npc_modelB_crops_debug.beton \
		--logging.folder $LOGGING_FOLDER/resnet50_crops/supervised_train_A_test_B_wd_0.1_lars 0 \
