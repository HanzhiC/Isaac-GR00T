# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import InvertibleModalityTransform
from gr00t.experiment.data_config import DATA_CONFIG_MAP

from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
from gr00t.data.dataset import LeRobotSingleDataset
import os
import gr00t


# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
EMBODIMENT_TAG = "gr1"
data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()


def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language


def build_eagle_processor(eagle_path: str):
    eagle_processor = AutoProcessor.from_pretrained(
        eagle_path, trust_remote_code=True, use_fast=True
    )
    eagle_processor.tokenizer.padding_side = "left"
    return eagle_processor


def collate(features: List[dict], eagle_processor):
    """
    Collate function that processes outputs from apply_vlm_processing.
    
    Args:
        features: List of dictionaries, each containing eagle_content from apply_vlm_processing
        eagle_processor: Eagle processor for handling text and images
        
    Returns:
        batch: Dictionary with processed eagle inputs and other batched tensors
    """
    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            # Collect text_list, image_inputs, and video_inputs from all features
            text_list = []
            image_inputs = []
            video_inputs = []
            
            for v in values:
                # Handle text_list
                if "text_list" in v and v["text_list"]:
                    text_list += v["text_list"]
                
                # Handle image_inputs
                if "image_inputs" in v and v["image_inputs"]:
                    image_inputs += v["image_inputs"]
                
                # Handle video_inputs
                if "video_inputs" in v and v["video_inputs"]:
                    video_inputs += v["video_inputs"]
            
            # Process with eagle_processor
            eagle_inputs = eagle_processor(
                text=text_list, 
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt", 
                padding=True
            )
            
            # Add eagle_ prefix to all keys
            for k, v in eagle_inputs.items():
                batch["eagle_" + k] = v
                
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
            
    return batch


def apply_vlm_processing(eagle_processor, batch):
    """
    Args:
        batch:
            video: [V, T, C, H, W]
    Returns: required input with the format `BatchFeature`
    """
    # TODO(YL, FH): check if this is correct
    images = batch["images"]  # [V, T, C, H, W]
    images.shape[0]

    np_images = rearrange(images, "v t c h w -> (t v) c h w")
    text_content = []

    # handle language
    lang = batch["language"]
    if isinstance(lang, list):
        lang = lang[0]

    text_content.append({"type": "text", "text": lang})

    eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
    eagle_image = [{"type": "image", "image": img} for img in eagle_images]
    eagle_conversation = [
        {
            "role": "user",
            "content": eagle_image + text_content,
        }
    ]

    text_list = [
        eagle_processor.apply_chat_template(
            eagle_conversation, tokenize=False, add_generation_prompt=True
        )
    ]
    breakpoint()
    image_inputs, video_inputs = eagle_processor.process_vision_info(eagle_conversation)
    eagle_content = {
        "image_inputs": image_inputs,
        "video_inputs": video_inputs,
        "text_list": text_list,
    }
    inputs = {}
    inputs["eagle_content"] = eagle_content
    return inputs

# Create the dataset
dataset = LeRobotSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=data_config.modality_config(),
    video_backend="decord",
    video_backend_kwargs=None,
    transforms=None,  # We'll handle transforms separately through the policy
    embodiment_tag=EMBODIMENT_TAG,
)

# Prepare the data for the VLM processing
step_data = dataset[0]
remapping = {
    "video.ego_view": "images", # uint8, [V, T, C, H, W]
    "annotation.human.action.task_description": "language", # [str]
}
step_data_new = {}  
for key, value in step_data.items():
    if key in remapping:
        step_data_new[remapping[key]] = value
# step_data_new["language"] = [formalize_language(l) for l in step_data_new["language"]]
step_data_new["images"] = rearrange(step_data_new["images"], "t h w c -> t c h w")[None]
step_data_new["language"] = np.array([step_data_new["language"][0]])

# Apply the VLM processing
eagle_processor: ProcessorMixin = build_eagle_processor(DEFAULT_EAGLE_PATH)
vlm_inputs = apply_vlm_processing(eagle_processor, step_data_new)
batch = collate([vlm_inputs], eagle_processor)
for k, v in batch.items():
    save_path = ".tmp/vlm_outputs/" + k + ".pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(v, save_path)

