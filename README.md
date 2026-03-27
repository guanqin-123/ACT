## settle

### 1. Docker Setup (Recommended)

Download the container:
```bash
wget https://zenodo.org/records/15301601/files/act.tar -O act.tar
```

Load the Docker image:
```bash
docker load -i act.tar
```

Start a container:
```bash
docker run -it --name act-artifact-container batchedfuzzing/act
```

Export the python environment:
```bash
export PYTHONPATH=/app:$PYTHONPATH
```
You will be placed in the our project ENV.

And all source code is under ```act```

If you have exited the container, you can restart it with:
```bash
docker start -ai act-artifact-container
```

Then follow the instructions in `README.md` (here, below)


### 2. Data Availability

All raw experiment data used by our runs is included in `experiments/`, which are used for all figures and tables to support our paper's claims and contribution.

```
experiments/
  rq1/                          # RQ1 (throughput/speedup) & RQ2 (cumulative violations)
    cifar100/                   #   3 methods × 1 run each
    tinyimagenet/               #     batch_aniso/run_0.json
    trafficsigns/               #     batch_iso/run_0.json
                                #     seq_fix/run_0.json
  rq3/                          # RQ3 (scale-factor sensitivity)
    0.01/                       #   6 scale factors: 0.01, 0.05, 0.10, 0.20, 0.30, 0.50
    0.05/                       #     each contains 2 benchmarks × 2 methods
    0.10/                       #       (batch_aniso, batch_iso)
    0.20/
    0.30/
    0.50/
  rq_batchsize/                 # RQ4 (batch-size significance)
    B_001/                      #   6 batch sizes: B_001, B_010, B_050, B_099, B_100, B_199
    B_010/                      #     each contains cifar100/ and/or tinyimagenet/
    B_050/                      #       with batch_aniso/ and batch_iso/ subdirs
    B_099/
    B_100/
    B_199/
```

Each `run_0.json` contains: `violations`, `coverage`, `ttfv`, `throughput`, `total_time`, `violation_timestamps`, and per-group breakdowns.


### 3.Scripts for Figures and Tables (Used in Paper)

All scripts are in `act/pipeline/script/`.

---

**3.1 RQ1 — Table 1 (throughput & speedup)**
```
python act/pipeline/script/results_aggregator.py experiments/rq1
```
```
Found 9 run(s) across 3 benchmark(s).

------------------------------------------------------------------------------------------------
Benchmark       Method               Violations      Coverage(%)        TTFV(s)       Thpt(it/s)
------------------------------------------------------------------------------------------------
trafficsigns    Seq-Fixed                 40622           100.00           62.6             75.0
                Batch-Iso                 73316           100.00            0.1            407.3
                Batch-Ani                 75208           100.00            0.1            417.9

cifar100        Seq-Fixed                 14562            57.19           60.1             15.2
                Batch-Iso                 60242            64.06            0.6            516.4
                Batch-Ani                 61527            64.07            0.6            524.0

tinyimagenet    Seq-Fixed                 16890            59.04           60.2             10.9
                Batch-Iso                 23125            70.04            1.2            395.1
                Batch-Ani                 25274            70.04            1.1            430.7
------------------------------------------------------------------------------------------------

LaTeX table written to figures/table1_latex.tex
```

---

**3.2 RQ2 — Figure 2 (cumulative violations over time)**
```
python act/pipeline/script/rq2_figure2_broken.py experiments/rq1 --output figures/figure2_broken
```
```
Generating broken-axis Figure 2: 3 benchmarks, break=60.0s
Saved: figures/figure2_broken.png
Saved: figures/cifar100.png
Saved: figures/tinyimagenet.png
Saved: figures/trafficsigns.png
```

---

**3.3 RQ2 — Batch advantage table**
```
python act/pipeline/script/rq2_table_advantage.py experiments/rq1 --output figures/table_advantage
```
```
Time                          trafficsigns                        cifar100                    tinyimagenet
----------------------------------------------------------------------------------------------------------
1 min                    +32,274 / +33,366               +43,109 / +42,382               +22,926 / +25,075
5 min                    +61,664 / +63,556               +58,759 / +60,044               +21,335 / +23,484
30 min                   +32,846 / +34,738               +53,364 / +54,649               +17,250 / +19,399
60 min                   +32,694 / +34,586               +50,129 / +51,414               +13,708 / +15,857

Saved: figures/table_advantage.tex
```

---

**3.4 RQ3 — Scale factor sensitivity table**
```
python act/pipeline/script/rq3_aggregator.py experiments/rq3
```
```
Found 24 run(s).

------------------------------------------------------------------------------------------------------
Benchmark       Method              s=0.01      s=0.05       s=0.1       s=0.2       s=0.3       s=0.5
------------------------------------------------------------------------------------------------------
Cifar100        Batch-Iso            52111       61195       60578       59476       56990       47887
                Batch-Aniso          56311       59522       59951       56748       47166       46741
------------------------------------------------------------------------------------------------------
TinyImageNet    Batch-Iso            16066       17354       22728       21907       22553       20232
                Batch-Aniso          20442       23905       23296       21324       18957       19838
------------------------------------------------------------------------------------------------------

LaTeX table written to figures/table3_rq3_scale.tex
```

---

**3.5 RQ4 — Batch size significance table**
```
python act/pipeline/script/rq4_batchsize_analysis.py experiments/rq_batchsize --merged
```
```
=== CIFAR-100 ===

--------------------------------------------------------------------------------
            Iso-1   Iso-10   Iso-50   Iso-99    Ani-1   Ani-10   Ani-50   Ani-99
--------------------------------------------------------------------------------
   Iso-1        —      ✗✗✗      ✗✗✗      ✗✗✗        ≡      ✗✗✗      ✗✗✗      ✗✗✗
  Iso-10      ✓✓✓        —       ✓✓      ✓✓✓      ✓✓✓        ≡       ✓✓      ✓✓✓
  Iso-50      ✓✓✓       ✗✗        —        ✓      ✓✓✓       ✗✗        ≡        ✓
  Iso-99      ✓✓✓      ✗✗✗        ✗        —      ✓✓✓      ✗✗✗        ✗        ≡
   Ani-1        ≡      ✗✗✗      ✗✗✗      ✗✗✗        —      ✗✗✗      ✗✗✗      ✗✗✗
  Ani-10      ✓✓✓        ≡       ✓✓      ✓✓✓      ✓✓✓        —       ✓✓      ✓✓✓
  Ani-50      ✓✓✓       ✗✗        ≡        ✓      ✓✓✓       ✗✗        —        ✓
  Ani-99      ✓✓✓      ✗✗✗        ✗        ≡      ✓✓✓      ✗✗✗        ✗        —
--------------------------------------------------------------------------------

=== TinyImageNet ===

--------------------------------------------------------------------------------------------------
            Iso-1   Iso-10   Iso-50  Iso-100  Iso-199    Ani-1   Ani-10   Ani-50  Ani-100  Ani-199
--------------------------------------------------------------------------------------------------
   Iso-1        —      ✗✗✗      ✗✗✗      ✗✗✗       ✗✗        ≡      ✗✗✗      ✗✗✗      ✗✗✗      ✗✗✗
  Iso-10      ✓✓✓        —       ✓✓      ✓✓✓      ✓✓✓      ✓✓✓        ≡       ✓✓      ✓✓✓      ✓✓✓
  Iso-50      ✓✓✓       ✗✗        —       ✓✓      ✓✓✓      ✓✓✓       ✗✗        ≡       ✓✓      ✓✓✓
 Iso-100      ✓✓✓      ✗✗✗       ✗✗        —        ✓      ✓✓✓       ✗✗       ✗✗        ≡        ✓
 Iso-199       ✓✓      ✗✗✗      ✗✗✗        ✗        —       ✓✓      ✗✗✗      ✗✗✗        ✗        ≡
   Ani-1        ≡      ✗✗✗      ✗✗✗      ✗✗✗       ✗✗        —      ✗✗✗      ✗✗✗      ✗✗✗      ✗✗✗
  Ani-10      ✓✓✓        ≡       ✓✓       ✓✓      ✓✓✓      ✓✓✓        —       ✓✓       ✓✓      ✓✓✓
  Ani-50      ✓✓✓       ✗✗        ≡       ✓✓      ✓✓✓      ✓✓✓       ✗✗        —       ✓✓      ✓✓✓
 Ani-100      ✓✓✓      ✗✗✗       ✗✗        ≡        ✓      ✓✓✓       ✗✗       ✗✗        —        ✓
 Ani-199      ✓✓✓      ✗✗✗      ✗✗✗        ✗        ≡      ✓✓✓      ✗✗✗      ✗✗✗        ✗        —
--------------------------------------------------------------------------------------------------
Merged LaTeX table written to figures/rq_batchsize_table_merged.tex
```



### 4.Experiment Reproduction from Scratch

All scripts are in `act/pipeline/script/`. Shared configuration is in `rq1_config.py` and `rq3_config.py`.

All experiments were run on a single NVIDIA RTX PRO 6000 Blackwell Max-Q GPU (${\approx}$96\,GB VRAM). Multi-GPU is not configured. CPU execution is also supported via `--device cpu` but will result in lower throughput and number of generation of the violations. 


---

**4.1 RQ1 & RQ2 — Batched Fuzzing Experiments**

Runs 3 benchmarks (TrafficSigns, CIFAR-100, TinyImageNet) × 3 methods (batch\_iso, batch\_aniso, seq\_fix) × 5 independent runs. Time budget: 60s per instance.

```bash
bash act/pipeline/script/run_rq1.sh
```

Expected time per config (1 run):
- `batch_iso` / `batch_aniso`: ~3 min (TrafficSigns, 3 model groups × 60s), ~2 min (CIFAR-100), ~1 min (TinyImageNet)
- `seq_fix`: ~44 min (TrafficSigns, 44 instances × 60s), ~3.3h (CIFAR-100/TinyImageNet, 199 instances × 60s)
- Total for all 5 runs (9 configs × 5): ~3 days

The script **resumes automatically** — already-completed runs are skipped. Results are saved to `experiments/rq1/{benchmark}/{method}/run_N.json`.


#### Trial for Validation

To save reviewers' time, we provide a short trial that runs all three methods on the TrafficSigns benchmark with 10 instances and 1 run each (~3 min total on GPU): 


```bash
bash act/pipeline/script/run_rq1.sh --benchmark trafficsigns --runs 1 --max-instances 10 --output-dir experiments/rq1_trial
```

Note: 10 instances are needed for batch methods to form meaningful groups and demonstrate throughput differences versus `seq_fix`.

Upon completion, the aggregated results are printed automatically:
```
------------------------------------------------------------------------------------------------
Benchmark       Method               Violations      Coverage(%)        TTFV(s)       Thpt(it/s)
------------------------------------------------------------------------------------------------
trafficsigns    Seq-Fixed                 17051           100.00           44.5            180.6
                Batch-Iso                 44706           100.00            0.1            744.7
                Batch-Ani                 60846           100.00            0.2           1013.4
------------------------------------------------------------------------------------------------
```

- The table shows that batch methods find significantly more violations (~3x) with ~5x higher throughput and near-instant TTFV compared to `seq_fix`. 
- The highly expected output is the number of violations dected by Seq-Fixed is less than the Batched-* variants. 





Available options:
```
--benchmark NAME      Run only one benchmark (trafficsigns / cifar100 / tinyimagenet)
--method NAME         Run only one method (batch_iso / batch_aniso / seq_fix)
--runs N              Runs per config (default: 5)
--timeout SECONDS     Per-instance timeout (default: 60)
--device DEVICE       cpu or cuda (default: cuda)
--output-dir DIR      Output directory (default: experiments/rq1)
--max-instances N     Limit instances per run
```

---

**4.2 RQ3 — Scale Factor Sensitivity**

Runs 2 benchmarks (CIFAR-100, TinyImageNet) × 6 scale factors (0.01, 0.05, 0.1, 0.2, 0.3, 0.5) × 2 methods (batch\_iso, batch\_aniso) × 5 independent runs = 120 configs. Time budget: 60s per model group.

```bash
bash act/pipeline/script/run_rq3.sh
```

Expected time: ~3 hours total on GPU (60s per model group, 120 configs).

The script **resumes automatically** — already-completed runs are skipped. Results are saved to `experiments/rq3/{scale}/{benchmark}/{method}/run_N.json`.

---

**4.3 RQ4 — Batch Size Significance**

RQ4 varies the batch size B across CIFAR-100 and TinyImageNet (90 configs total). Pre-computed results are already included in `experiments/rq_batchsize/`.

To reproduce from scratch:

```bash
bash act/pipeline/script/rq_batchsize_runner.sh
```

Expected time: ~1.5 hours (90 configs × 60s each). The script **resumes automatically**. Results are saved to `experiments/rq_batchsize/B_{N}/{benchmark}/{method}/run_N.json`.

Available options:
```
--device DEVICE    cpu or cuda (default: cuda)
--dry-run          Preview without executing
```

### License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
