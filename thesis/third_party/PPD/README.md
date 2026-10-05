# Post-hoc Popularity Bias Correction in GNN-based Collaborative Filtering (ACM Web Conference 2026) 
## Paper link: [PDF](https://dl.acm.org/doi/epdf/10.1145/3774904.3792670)

## Source code of PPD

Options to reproduce results reported in the paper:

For KuaiRec dataset, use following command to run it:
```bash
1. cd LightGCN/code 
2. pip install -r requirements.txt
3. sh run.sh KuaiRec lgn lgn 2 311 0.20 0.50
```

For Coat dataset, go to the root directory of this project, and then use the following command:
```bash
1. cd AdvInfoNCE
2. python setup.py build_ext --inplace
3. pip install -r requirements.txt
4. python main.py --train_norm --pred_norm --modeltype LGN --dataset Coat --n_layers 2 --batch_size 2048 --neg_sample 1 --lr 5e-5 --dsc sota_Coat_LGN_PPD --patience 10 --embed_size 256  --use_ppd --regs 0.0 --beta_ppd 0.10 --phi_ppd 1.0 
```

For Yahoo dataset, go to the root directory of this project, and then use the following command:
```bash
1. cd AdvInfoNCE
2. python main.py --modeltype LGN --dataset yahoo.new --n_layers 2 --batch_size 2048 --neg_sample 1 --lr 5e-5 --dsc sota_Yahoo_LGN_PPD --patience 5 --embed_size 256 --use_ppd  --beta_ppd 0.30 --phi_ppd 1.0
```

For running PPD with the SGL backbone, stay in the AdvInfoNCE directory.
```bash
1. python main.py --modeltype SGL --dataset KuaiRec --n_layers 2 --batch_size 2048 --neg_sample 1 --lr 5e-3 --dsc sota_KuaiRec_PPD_SGL --patience 10 --embed_size 256 --use_ppd --beta_ppd 0.20 --phi_ppd 0.5
2. python main.py --train_norm --pred_norm --modeltype SGL --dataset Coat --n_layers 2 --batch_size 2048 --neg_sample 1 --lr 5e-5 --dsc sota_Coat_PPD_SGL --patience 10 --embed_size 256  --use_ppd --regs 0.0 --beta_ppd 0.20 --phi_ppd 1.0
3. python main.py --modeltype SGL --dataset yahoo.new --n_layers 2 --batch_size 2048 --neg_sample 1 --lr 5e-5 --dsc sota_Yahoo_PPD_SGL --patience 5 --embed_size 256 --use_ppd --train_norm --pred_norm --beta_ppd 0.30 --phi_ppd 1.0
```

If you have any problem, please feel free to send me an email mislam34@uic.edu.

Please note that, in our work, we have used and modified parts of the codebase from previous works, with the corresponding references provided below.
## References
1. Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 639–648.
2. Huachi Zhou, Hao Chen, Junnan Dong, Daochen Zha, Chuang Zhou, and Xiao Huang. 2023. Adaptive popularity debiasing aggregator for graph collaborative filtering. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 7–17.
3. An Zhang, Leheng Sheng, Zhibo Cai, Xiang Wang, and Tat-Seng Chua. 2023. Empowering collaborative filtering with principled adversarial contrastive loss. Advances in Neural Information Processing Systems 36 (2023), 6242–6266.