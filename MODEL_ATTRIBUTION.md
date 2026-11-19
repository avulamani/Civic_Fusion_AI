# Computer-vision model attribution

## Selected model: YOLO-World v2

The project exports `yolov8s-worldv2.pt` with the prompt list in `vision_config.py` to create `models/municipal_worldv2.onnx`. YOLO-World is an open-vocabulary detector developed by Tencent AI Lab and integrated with Ultralytics. The dashboard prompts it for 10 municipal issue categories and treats its outputs as zero-shot suggestions, not validated municipal predictions.

- Ultralytics YOLO-World documentation and model/export support: <https://docs.ultralytics.com/models/yolo-world>
- Original YOLO-World project: <https://github.com/AILab-CVC/YOLO-World>
- Ultralytics licensing information: <https://docs.ultralytics.com/>

Ultralytics states that its code and models are offered under AGPL-3.0 or a separate Enterprise license. This is an academic project; review and comply with the applicable terms before distributing a deployment or model weights.

## Original fallback checkpoint

`models/municipal_issues.pt` is from [Ank0805/waste-pothole-manhole-detection](https://huggingface.co/Ank0805/waste-pothole-manhole-detection), whose model card declares an MIT license. Its classes are `waste_pile`, `pothole`, and `open_manhole`. It is retained as the original starter artifact but is not used by the new `vision.py` inference path.

## Evaluation note

Neither zero-shot prompt confidence nor NLP classifier confidence is measured accuracy. Evaluate the exported detector on a separate labeled image set before reporting precision, recall, or mAP. Several target conditions (such as a non-working lamp or a hidden pipe leak) may not be observable in a single photo; preserve the NLP result and flag uncertain image results for human review.
