# [NeurIPS 2025] TSP
Official PyTorch implementation for paper "How Does Topology Bias Distort Message Passing in Graph Recommender? A Dirichlet Energy Perspective".
## TODOs
- [x] arXiv paper link
- [x] camera-ready paper link
- [x] training and inference code

## Files

| File | Content |
|---|---|
| `utils.py` | sparse helpers (compatible with the existing LightGCN code), Recall@k / NDCG@k evaluation (overall + tail) |
| `dataloader.py` | the datasets: Adressa, Gowalla, Yelp2018, ML10M, Globo |
| `model.py` | LightGCN backbone (BPR training) |
| `tsp.py` | Our proposed method |
| `main.py` | training + test-time TSP + evaluation entry point |

## Data

Put real data under `/data/<dataset>/` (dataset ∈ {`adressa`, `gowalla`,
`yelp2018`, `ml10m`, `globo`}) as either:

- `interactions.txt` — one `user_id item_id` pair per line; the loader applies
  the **unbiased 80/10/10 uniform split** of MACR used in the paper, or
- pre-split `train.txt` / `valid.txt` / `test.txt` (same format; LightGCN-style
  `u i1 i2 ...` lines also work).

Dataset sources: Adressa (<https://reclab.idi.ntnu.no/dataset>), Gowalla
(<https://snap.stanford.edu/data/loc-gowalla.html>), Yelp2018
(<https://www.yelp.com/dataset>, 10-core), Globo
(<https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom>),
ML10M (<https://grouplens.org/datasets/movielens/10m>, ratings binarized).

## Run

```bash
python main.py --dataset gowalla --K 3 --beta 1.0 --L 2 --theta 0.3 --topk 10
```


## Citation
If you find this repository helpful, please consider citing:
```bibtex
@inproceedings{ji2025doestopologybiasdistort,
      title={How Does Topology Bias Distort Message Passing? A Dirichlet Energy Perspective}, 
      author={Yanbiao Ji and Yue Ding and Dan Luo and Chang Liu and Yuxiang Lu and Xin Xin and Hongtao Lu},
      booktitle={Advances in Neural Information Processing Systems 38: Annual Conference
                  on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego,
                  CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December
                  5, 2025},
      year= {2025},
}
```
