# Chạy đầy đủ PFGR-Lite từ R0 đến R10

Entry point chuẩn cho yêu cầu **FULL đến R10** là `scripts/run_pfgr_full_pipeline.py`.
`run_pfgr_r4_evidence.py` vẫn giữ chế độ tương thích chỉ đến R4B.
FULL ở đây là phạm vi thí nghiệm R0–R10; số epoch và số subject bên dưới là
ngân sách **PROVISIONAL**, không phải bằng chứng hội tụ hay cấu hình paper đã duyệt.

## Một lệnh trên server

Chạy từ repo với môi trường Python/PyTorch và dữ liệu đã chuẩn bị. Thay ba đường
dẫn `/srv/...` bằng BraTS thô, MedicalNet ResNet10 và split JSON đã kiểm tra của bạn.
SHA256 được tính từ đúng file checkpoint; runner không tải weights hay cài dependency.

```bash
rtk proxy python scripts/run_pfgr_full_pipeline.py \
  --data-root '/srv/data/BraTS2021_Training_Data' \
  --medicalnet-checkpoint '/srv/checkpoints/resnet_10.pth' \
  --medicalnet-sha256 "$(rtk proxy python -c 'import hashlib; from pathlib import Path; print(hashlib.sha256(Path("/srv/checkpoints/resnet_10.pth").read_bytes()).hexdigest())')" \
  --split-file '/srv/data/brats21_split.json' \
  --run-root '/srv/runs/pfgr-full' \
  --device cuda:0 --profile provisional \
  --static-epochs 20 --updater-epochs 20 --value-epochs 50 \
  --train-max-subjects 128 \
  --bank-max-subjects 64 --bank-max-states 3 --bank-candidate-count 32 \
  --bank-replay-count 8 --value-batch-size 32 --value-learning-rate 0.001 \
  --calibration-max-subjects 128 --evaluation-max-subjects 32 \
  --teacher-query-count 1024 --candidate-chunk-size 1 --decode-chunk-size 1024 \
  --package-max-file-mib 64 --package-max-archive-mib 256 \
  --seed 20260907
```

Thêm `--plan-only` để chỉ kiểm tra input/hash/config/cờ CLI và lưu argv kế hoạch;
không chạy CLI, không đọc volume MRI và không train. File server chưa có ở máy lập
kế hoạch được ghi là `unavailable_plan_only`; config cục bộ vẫn phải hợp lệ.
Mỗi lần gọi tạo thư mục mới, kể cả plan-only. Có thể dùng `--run-name ten-moi`;
runner từ chối thư mục đã tồn tại, không ghi đè checkpoint/log cũ.

Nếu đã có role manifest, thêm `--roles-file '/srv/data/pfgr_roles.json'`. Nếu chưa,
R0 tạo manifest bằng logic hiện có. Producer-fit, calibration-fit,
calibration-allowance, validation và test được kiểm tra bởi CLI; runner không tự
đổi assignment, không chọn subject theo kết quả. `--python /path/to/python` chọn
interpreter cho từng process con. `--stage-timeout-seconds` đặt timeout tùy chọn;
mặc định không giới hạn thời gian một stage.

## DAG thực thi

| R | Công việc và artifact chính |
|---|---|
| R0–R2 | Preflight, runbook-check, synthetic/real smoke và benchmark; benchmark bật mặc định. |
| R3 | Train B0/B1/B2/B-light bằng ngân sách khai báo; đánh giá 4 validation subject theo kế hoạch R4 tương thích. |
| R4 | Hai updater U-only/U+spectral từ cùng B2; R4B headroom diagnostic 4 validation subject, không biến kết quả thành giấy duyệt. |
| R5 | `bank-build --engineering-only` trên U+spectral/B2, rồi replay verify; index thật tại `R5-bank/s2/bank/index.json`. |
| R6 | Fit/evaluate V126, V270, V366, V222 trên cùng index; join immutable row/action/context/hash/gain-scale; chọn trước V366 cho downstream. |
| R7 | Thu thập và fit calibration trên role train tách biệt; `--engineering-only --exploratory-run` ghi `execution_authorization.json` thực. |
| R8 | Validation: noop/static K0; random/fixed_learned/parallel_topk K1/2/4; adaptive K1/2/4 chỉ khi có artifact adaptive hợp lệ. |
| R9 | Test: cùng matrix policy/K và seed 17/29/41 đã khai báo; không chọn winner bằng test; ghi authorization thực cho từng command. |
| R10 | Interrupted/resumed/uninterrupted synthetic S0 và cached V, so cursor/weights/optimizer/RNG thực trong process riêng; đóng gói evidence. |

Các learned controls dùng V366 đã khai báo trước, không dùng best V chọn
theo validation/test. Ba seed control là seed policy/randomization, không phải
ba lần train độc lập: checkpoint B/U/V chỉ có seed train được khai báo.

Chế độ exploratory hiện tại luôn giữ calibration ở capability `diagnostic`
(hoặc trả dữ liệu không đủ), kể cả khi cohort đủ lớn: không xuất `adaptive.pt`. Runner giữ nguyên a/b/allowance/counts/counters đo
được, ghi adaptive `NOT_AVAILABLE` kèm lý do, bỏ riêng 18 branch adaptive và vẫn
chạy control R8/R9 cùng R10. Lỗi process, artifact thiếu trái contract hoặc
calibration hỏng là `ERROR`, không được đổi thành diagnostic hợp lệ. Không tạo
`APPROVED`, `HEADROOM_CONFIRMED`, review receipt giả hay xác nhận scientific PASS.

## Ngân sách và resume

`--static-epochs`, `--updater-epochs`, `--value-epochs` phải khai báo rõ.
`--static-max-steps`/`--updater-max-steps` là cap tùy chọn; không truyền thì không
cắt train thành smoke hai bước. Không truyền `--train-max-subjects` thì dùng
producer-fit cohort theo CLI. R5 expose số subject/state/candidate; teacher
query-count là số draw IID, không phải số voxel distinct được đảm bảo.

R10 mechanics luôn tách biệt với ngân sách train: S0 epochs=2, subjects=2,
batch=1, dừng ở committed update 1 rồi resume đến 2 và so với chạy mới đến 2.
Cached V dùng epochs=3, cùng bank/V366/seed/batch/lr, giới hạn 1→2 so với 2;
checkpoint vẫn incomplete có chủ đích để kiểm tra continuation, không được công
bố thành ValueArtifact đã fit xong. Helper yêu cầu cursor thực `[1,2,2]`, có
weights thay đổi sau resume, và tensor/optimizer/cursor/RNG khớp tuyệt đối với
reference. Thêm `--real-resume-check` để chạy thêm một probe S0 thật cùng cơ chế;
không dùng probe này thay train R3/R4.

## Đọc kết quả

Trong thư mục mới của lần chạy:

- `metadata/pipeline_manifest.json`: toàn bộ argv, budget, config và input hash,
  predecessor được khai báo, phạm vi EXPLORATORY và lựa chọn B2/U+spectral/V366.
- `metadata/pipeline_summary.json` và `pipeline_metrics.csv`: trạng thái từng
  stage, lỗi, cohort quan sát, metric nguồn và đường dẫn JSON chính xác. Có V
  MSE/rank/regret/sign/constant controls, calibration, policy K/STOP/action và
  MAE/PSNR/SSIM khi artifact nguồn có phép đo; thiếu hoặc nonfinite là `null`.
- `metadata/pipeline_events.jsonl`, `metadata/stage_logs/*/{command,exit}.json`
  và stdout/stderr: thời gian, exit code, argv dạng list và hash dependency thật.
- `stages/R6-same-bank-join/value_join.json`: join immutable row giữa 4 V; đây
  là cùng bank đã fit, không phải đánh giá khả năng tổng quát hóa V trên held-out.
- `metadata/control_join.json`: paired subject với cùng role/seed/context/Z0,
  improvement trên noop; MAE/Charbonnier dùng noop−policy, PSNR/SSIM dùng chiều
  ngược lại. Latency lấy nguyên `pipeline_elapsed_seconds`, bao gồm việc đo
  metric/teacher sau prediction, không tự gán là deployment latency. Frontier
  tổng hợp để `null`; không xếp winner từ test.
- `stages/R10-*/resume_comparison.json`: so tensor thật và committed cursor.
- `package/evidence/evidence.zip`, `package/manifest.json`: evidence được
  allowlist, SHA256 thật, không chứa volume MRI hay checkpoint `.pt`.

`action_metrics.jsonl` gốc và các trace/metric được giữ tại stage; không ghi đè
bằng các số tổng hợp. Artifact nào vượt giới hạn gói hoặc bị loại được ghi rõ
trong manifest package. Runner dùng một nguồn stage-tree đệ quy để argv package
vẫn gọn cho toàn bộ 122 stage; log package nằm ngoài nguồn đang đóng gói.

Ví dụ trên đặt giới hạn 64 MiB/file và 256 MiB/ZIP vì bảng tổng hợp nhiều policy
và 32 subject có thể lớn hơn mức mặc định 8/64 MiB. Đây là giới hạn cấu hình,
không phải kích thước đã đo của một run MRI. File được allowlist mà vượt giới
hạn sẽ làm package báo lỗi; không âm thầm cắt metric để tạo ZIP thành công.

Exit 0 có thể là `completed` hoặc `completed_with_unavailable`; nó chỉ xác nhận
DAG phần mềm hoàn tất theo các capability thực. `error`/exit 1 nghĩa là có
stage/package lỗi hoặc dependency bị chặn; stage độc lập vẫn được thử và evidence
lỗi vẫn được đóng gói. R4B inconclusive không tự chặn R5 engineering. Ctrl-C dừng
việc khởi chạy stage mới, vẫn thử đóng gói, rồi trả 130.

## Verification cục bộ

Chưa chạy MRI thật, train đầy đủ, CUDA hay server evaluation trong vòng sửa này.
Root kiểm lại combined current tree sau khi các thay đổi code hoàn tất:
`python -m pytest -q tests/features/point_guided/pfgr_lite tests/features/point_guided/test_frontend_forward.py`
đạt **521 passed, 1 skipped trong 141.13 s**. Ca skip cần CUDA. Không cộng các
focused test totals bên dưới vào con số này.
Kiểm tra ngày 2026-09-12:

- `rtk proxy python -m pytest -q tests/features/point_guided/pfgr_lite/test_full_pipeline_runner.py tests/features/point_guided/pfgr_lite/test_pipeline_runner.py`: **35 passed** (46.67 s); mock compute, DAG đầy đủ, no-overwrite/argv/cohort/hash/failure isolation, cùng packager thật.
- `rtk proxy python -m pytest -q tests/features/point_guided/test_frontend_forward.py`: **24 passed** (6.50 s).
- `compileall` trên hai runner và test mới, Ruff check, `git diff --check`: pass.
- Probe CLI synthetic S0 thật: interrupted/resumed/uninterrupted và process helper
  đọc checkpoint thật, cursor `[1,2,2]`, **339 tensor comparisons**, **62 tensors
  thay đổi sau resume**, không khác reference. Artifact tạm trong
  `.pytest_cache/full-r10-tdf7jtf6/stages/R10-synthetic-comparison/resume_comparison.json`.
- Probe CLI cached V thật trên bank fixture synthetic bất biến cùng role của
  checkpoint: cursor `[1,2,2]`, **25 tensor comparisons**, **6 tensors thay đổi**,
  optimizer/cursor/RNG/weights bằng tuyệt đối. Artifact tạm trong
  `.pytest_cache/full-v-resume-xgwnl_2x/comparison/resume_comparison.json`.
- Packager thật đã đóng gói hai probe trên thành **42 archive entries**, chứa cả
  hai comparison và ba `value_fit_incomplete.json`, không có `.pt`; archive
  SHA256 `0575a201b9346afbf607467663312f6b76bcad4861b42be63c3265773fdcd680`,
  tại `.pytest_cache/full-actual-package-8s88oo6k/evidence/evidence.zip`.
  Manifest này giữ `evidence_status=MISSING_REQUIRED`: hai probe R10 không có
  đủ mọi slot evidence tổng quát mà packager kỳ vọng cho từng run, như
  effective policy/source/environment/weights; archive integrity không đồng
  nghĩa toàn bộ evidence nghiên cứu đã đầy đủ.

Các đường dẫn `.pytest_cache` là evidence tạm cục bộ, có thể bị xóa khi dọn cache.
Những probe trên xác minh cơ chế software; chúng không phải thực nghiệm FULL
trên MRI thật hoặc kết quả chất lượng của model đã train.
