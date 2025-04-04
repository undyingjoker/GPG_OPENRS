<div align="center">
  <h1 align="center">GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning</h1>
  <div align="center">
        <div class="is-size-5 publication-authors">
              <span class="author-block"> <a href="https://scholar.google.com/citations?user=jn21pUsAAAAJ" target="_blank">Xiangxiang Chu</a>, </span>
              <span class="author-block"> <a href="https://scholar.google.com/citations?user=X0o0Ib8AAAAJ" target="_blank">Hailang Huang</a>, </span>
              <span class="author-block">Xiao Zhang, </span>
              <span class="author-block">Fei Wei, </span>
              <span class="author-block"> <a href="https://www.semanticscholar.org/author/Yong-Wang/1683878" target="_blank">Yong Wang</a></span>
        </div>
        <div class="is-size-5 publication-authors">
              <span class="author-block">AMAP, Alibaba Group</span>
        </div>
    </div>
  📖<a href="https://arxiv.org/abs/2504.02546">Paper</a> | Work in progress.</h3>
</div>
<div>
Reinforcement Learning (RL) can directly enhance the reasoning capabilities of large language models without extensive reliance on Supervised Fine-Tuning (SFT). In this work, we revisit the traditional Policy Gradient (PG) mechanism and propose a minimalist RL approach termed Group Policy Gradient (GPG). Unlike conventional methods, GPG directly optimize the original RL objective, thus obviating the need for surrogate loss functions. As illustrated in the figure below, by eliminating both the critic and reference models, and avoiding KL divergence constraints, our approach significantly simplifies the training process when compared to Group Relative Policy Optimization (GRPO). Our approach achieves superior performance without relying on auxiliary techniques or adjustments. Extensive experiments demonstrate that our method not only reduces computational costs but also consistently outperforms GRPO across various unimodal and multimodal tasks. 
    <img src="docs/images/figure0.svg" alt="GPG" width="100%">
</div>

------


# Usage

## Environment Installation

Clone this repository.

```bash
git clone git@github.com:AMAP-ML/GPG.git

cd GPG
```
Follow the repositories you need and install the required packages.

## Experiments on open-rs

Please refer to the training script: [`./open-rs/train.sh`](./open-rs/train.sh), [`./open-rs/recipes`](./open-rs/recipes)


## Experiments on VisualThinker-R1-Zero

Please refer to the training script: [`./VisualThinker-R1-Zero/src/open-r1-multimodal/run_grpo_SAT.sh`](./VisualThinker-R1-Zero/src/open-r1-multimodal/run_grpo_SAT.sh)


## Experiments on Visual-RFT

Please refer to the training script: [`./Visual-RFT/src/scripts/`](./Visual-RFT/src/scripts/)


## Experiments on R1-V

Please refer to the training script: [`./R1-V/src/scripts/run_grpo_GEOQA_qwen2.5_3b.sh`](./R1-V/src/scripts/run_grpo_GEOQA_qwen2.5_3b.sh)


## Q&A
If you have any questions, please submit an [issue](https://github.com/AMAP-ML/GPG/issues/new) or contact huanghailang.hhl\<AT\>alibaba-inc.com.


# Citation

If you find GPG or code useful, please cite

```bibtex
@misc{chu2025GPG,
      title={GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning}, 
      author={Xiangxiang Chu and Hailang Huang and Xiao Zhang and Fei Wei and Yong Wang},
      year={2025},
      eprint={2504.02546},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.02546}, 
}
```