# Acknowledgments and Third-Party Licenses

This HPE iOS application incorporates several open-source and research models. Below are the full licensing details.

---

## 1. GVHMR (Gravity-aware Video Human Motion Recovery)

**Authors:** 3D Vision Group, State Key Lab of CAD&CG, Zhejiang University  
**Reference:** https://github.com/zju3dv/GVHMR

### License

```
Copyright 2022-2023 3D Vision Group at the State Key Lab of CAD&CG,
Zhejiang University. All Rights Reserved.

Permission to use, copy, modify and distribute this software and its 
documentation for educational, research and non-profit purposes only.

Any modification based on this work must be open-source and prohibited 
for commercial use.

You must retain, in the source form of any derivative works that you
distribute, all copyright, patent, trademark, and attribution notices
from the source form of this work.

For commercial uses of this software, please send email to xwzhou@zju.edu.cn
```

### Citation

If you use GVHMR in your research or application, please cite the corresponding publications listed at https://github.com/zju3dv/GVHMR

---

## 2. ViTPose (Vision Transformer Pose Estimation)

**Authors:** ViTAE-Transformer Project  
**Reference:** https://github.com/ViTAE-Transformer/ViTPose

### License: Apache 2.0

```
Copyright 2021 ViTAE-Transformer Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Citation

Please cite the original ViTPose paper when using this component in research.

---

## 3. SMPL (Skinned Multi-Person Linear Body Model)

**Authors:** Max Planck Institute for Intelligent Systems  
**Reference:** https://smpl.is.tue.mpg.de/

### License

```
SMPL Model Download Terms of Use

The SMPL body model is released for research and educational purposes only.

Permission is granted to use the SMPL body model in any research or 
educational work where you:

1. Do not commercialize the SMPL model or any modification of it;
2. Respect the intellectual property of the authors;
3. Cite the relevant publications in any work that uses the model;
4. Include this notice in any source or derivative code.

For commercial licensing inquiries, contact: smpl@tue.mpg.de
```

### Citation

Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., & Black, M. J. (2015). 
SMPL: A Skinned Multi-Person Linear Model. 
ACM Transactions on Graphics (TOG), 34(6).

---

## 4. YOLOv8 / YOLOPose

**Authors:** Ultralytics  
**Reference:** https://github.com/ultralytics/ultralytics

### License: AGPL-3.0

```
Copyright (C) Ultralytics

YOLOv8 is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

You are free to use, modify, and distribute this software, provided that:
1. Any software that includes or uses YOLOv8 must also be released under AGPL-3.0;
2. You must provide users access to the source code;
3. You include a copy of this license with your distribution.

For commercial use or alternative licensing, contact: https://www.ultralytics.com/
```

### Citation

Ultralytics YOLOv8 Release Notes: https://github.com/ultralytics/ultralytics/releases

---

## 5. PyTorch and TorchVision

**Authors:** Meta AI Research  
**Reference:** https://pytorch.org/, https://github.com/pytorch/vision

### License: BSD 3-Clause

```
Copyright (c) 2016- Facebook, Inc (Adam Paszke)
and contributors

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of Facebook nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

## 6. CoreML and Apple Frameworks

Apple Vision framework, CoreML, and related components are provided under 
Apple's Software License Agreement.

---

## Summary of Obligations

| Component | Type | License | Compliance Required |
|-----------|------|---------|---------------------|
| GVHMR | Model | Research | Must retain attribution; cannot use commercially; contact authors for commercial use |
| ViTPose | Model | Apache 2.0 | Include Apache 2.0 license; can use commercially with conditions |
| SMPL | Model | Research | Must cite papers; no commercial use without licensing |
| YOLOv8 | Model | AGPL-3.0 | App must be open-source under AGPL-3.0 if distributed; OR use alternative license |
| PyTorch/TorchVision | Library | BSD 3-Clause | Include BSD license notice |
| Apple Frameworks | Library | Apple License | Follow Apple's terms |

---

## Important Notes

### COMMERCIAL USE RESTRICTION

Several models (GVHMR, SMPL, YOLOv8-AGPL) have restrictions on commercial use. 
If you plan to commercialize this app, you must:

1. **Contact GVHMR authors** at xwzhou@zju.edu.cn for GVHMR commercial licensing
2. **Obtain SMPL commercial license** from Max Planck Institute (smpl@tue.mpg.de)
3. **Either:** 
   - Replace YOLOv8 with a permissively licensed 2D pose detector (e.g., switch to Apple Vision framework only)
   - **OR** Release your entire app under AGPL-3.0 (requires open-sourcing and passing AGPL obligations to users)

### For Research/Educational Use

This app is fully compliant with all third-party licenses for:
- Academic research
- Educational purposes
- Non-profit use
- Open-source projects

---

## How to Contribute

Contributions are welcome! When you contribute, you agree to license your 
contributions under the terms of this project's license (MIT for app code,
with compatible third-party model licenses maintained).

---

## Questions?

For licensing questions or concerns, please:
1. Review the individual model pages linked above
2. Check the original GitHub repositories
3. Contact the model authors directly

---

**Last Updated:** 2026  
**Version:** 1.0
