你现在的 `configs/mvp.yaml` 主要改成了：

```text
train_samples: 1000
val_samples: 200
test_samples: 6200
epochs: 20
batch_size: 16
```

`Nx/Ny/n_active/gpt2_layers` 没变，所以不用改代码，直接重新训练和评估即可。

建议步骤如下。

**1. 先跑测试确认代码状态**
```powershell
uv run --with torch --with pytest --with numpy --with pyyaml --with transformers --with peft python -m pytest -q
```

**2. 如果继续用 `outputs/mvp`，先清理旧结果**
这会覆盖上一轮 MVP 结果。保守做法是先删旧产物：

```powershell
Remove-Item -LiteralPath outputs\mvp\proposed.pt,outputs\mvp\results.csv,outputs\mvp\train_history.csv -Force
```

如果你想保留上一轮结果，更推荐把 `configs/mvp.yaml` 里的：

```yaml
output_dir: outputs/mvp
```

改成类似：

```yaml
output_dir: outputs/mvp_train1000_test6200_e20
```

**3. 重新训练**
```powershell
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/train_mvp.py --config configs/mvp.yaml
```

训练完成后应生成：

```text
outputs/mvp/proposed.pt
outputs/mvp/train_history.csv
```

**4. 重新评估**
```powershell
uv run --with torch --with numpy --with pyyaml --with transformers --with peft python scripts/evaluate_mvp.py --config configs/mvp.yaml --checkpoint outputs/mvp/proposed.pt
```

评估完成后应生成：

```text
outputs/mvp/results.csv
```

**5. 检查结果**
```powershell
python -c "import csv; print(list(csv.DictReader(open('outputs/mvp/train_history.csv', newline='')))); print(list(csv.DictReader(open('outputs/mvp/results.csv', newline=''))))"
```

**6. 如果要记录这次实验**
```powershell
git add configs/mvp.yaml outputs/mvp/results.csv outputs/mvp/train_history.csv
git commit -m "exp: rerun mvp with larger dataset"
```

不要提交 `outputs/mvp/proposed.pt`，它已经被 `.gitignore` 排除了，体积很大。

注意：你现在 `test_samples=6200` 比训练集还大，评估会明显变慢，但不影响流程。如果只是想快速看训练改善，建议先用 `test_samples=200` 或 `1000`。