# Concept Selection Model
This is the code repository of "Understanding Multimodal Deep Neural Networks: A Concept Selection View". The paper is accepted by [CogSci 2024](https://escholarship.org/uc/item/1h67z0ww).
![image](https://github.com/user-attachments/assets/b5ce4b98-19be-4053-b299-2560a84388ec)

## Setup
- Use ```helper/prepare_concept_bank.ipynb``` to establish the concept library.
- Use ```helper/image_representation.py``` to get the image representations.
- Use ```helper/clip_label.py``` to annote the concepts by CLIP.

## Run
- Use ```rough_selection.ipynb``` to conduct the greedy rough selection.

- To run experiments, you can refer the command:
```
bash scripts/example.sh
```
```--algorithm``` can be chosen from from ```lp```, ```cbm``` and ```mask```.

- Use ```fine_selection.ipynb``` to conduct the mask fine selection.

## Citation
If you find this code useful, please consider citing our paper:
```
@inproceedings{shang2023understanding,
  title={Understanding Multimodal Deep Neural Networks: A Concept Selection View},
  author={Shang, Chenming and Zhang, Hengyuan and Wen, Hao and Yang, Yujiu},
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society},
  volume={46},
  year={2023}
}
```
