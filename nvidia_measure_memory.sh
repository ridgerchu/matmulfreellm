#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

nvidia-smi --query-gpu=memory.used --format=csv -lms 500
