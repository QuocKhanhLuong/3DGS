# Chạy PFGR R0–R4B và gửi lại một ZIP bằng chứng

**Muốn chạy đến R cuối (R10):** dùng
[hướng dẫn full pipeline](RUN_PFGR_FULL_PIPELINE.md) và
`scripts/run_pfgr_full_pipeline.py`. Script trong trang này chỉ giữ mục đích
chạy riêng R0–R4B để tương thích các lệnh trước.

Entrypoint: `scripts/run_pfgr_r4_evidence.py`. Script dùng Python standard
library để điều phối CLI hiện có, ghi log và gom metric. Môi trường server
phải có sẵn dependencies của repository, BraTS21 thật, checkpoint MedicalNet
cục bộ cùng SHA256 mong đợi, split đã review và ít nhất bốn subject validation.
Script không cài dependencies, tải checkpoint, sửa split hay chạy R5.

## Một lệnh chạy trên server

Đứng tại repository đã cập nhật code. Đặt các biến sau thành giá trị thật:

Nếu server không có `rtk`, bỏ đúng tiền tố `rtk proxy` ở lệnh dưới;
runner không phụ thuộc RTK.

| Biến | Ý nghĩa |
| --- | --- |
| `POINT_GUIDED_PYTHON` | Python executable trong môi trường server có torch/nibabel và dependencies repository |
| `BRATS21_ROOT` | Thư mục BraTS21 mà CLI hiện tại đọc được, chứa T1/T2/FLAIR và target T1ce |
| `MEDICALNET_CKPT` | Checkpoint MedicalNet có sẵn trên server |
| `MEDICALNET_SHA256` | SHA256 đã biết của checkpoint; phải khớp bytes thực tế |
| `BASELINE_SPLIT` | JSON split gốc đã review; không tự chia lại |
| `PFGR_DEVICE` | Thiết bị thực, ví dụ `cuda:0` hoặc `cpu` |
| `PFGR_RUN_ROOT` | Thư mục cha lưu run; mỗi lần chạy tạo một thư mục con mới |
| `PFGR_STATIC_EPOCHS` | Ngân sách epoch cho từng static arm |
| `PFGR_UPDATER_EPOCHS` | Ngân sách epoch cho từng updater arm |
| `PFGR_TRAINING_SEED` | Seed huấn luyện/khởi tạo tường minh |

```bash
rtk proxy "$POINT_GUIDED_PYTHON" scripts/run_pfgr_r4_evidence.py \
  --python "$POINT_GUIDED_PYTHON" \
  --data-root "${BRATS21_ROOT:?required}" \
  --medicalnet-checkpoint "${MEDICALNET_CKPT:?required}" \
  --medicalnet-sha256 "${MEDICALNET_SHA256:?required}" \
  --split-file "${BASELINE_SPLIT:?required}" \
  --device "${PFGR_DEVICE:?required}" \
  --run-root "${PFGR_RUN_ROOT:?required}" \
  --profile provisional \
  --static-epochs "${PFGR_STATIC_EPOCHS:?required}" \
  --updater-epochs "${PFGR_UPDATER_EPOCHS:?required}" \
  --seed "${PFGR_TRAINING_SEED:?required}" \
  --benchmark
```

Nếu đã có roles đã review, thêm `--roles-file "$PFGR_ROLES"`; nếu không, R0
tạo roles bằng chính CLI và tất cả stage sau dùng duy nhất file đó. Không dùng
held-out test để chọn checkpoint. CLI nạp dữ liệu và bảo vệ target boundary;
runner không tự đọc tensor hoặc tính loss.

Chạy thử kế hoạch: thêm `--plan-only` vào lệnh trên. Chế độ này kiểm config
cục bộ và cờ của parser hiện tại, lưu manifest/argv rồi thoát, không gọi CLI,
không train, không đánh giá và không đóng ZIP. Đường dẫn dữ liệu/checkpoint/
split/Python chỉ tồn tại trên server được ghi `unavailable_plan_only`, hash
không biết là `null`; nếu checkpoint có sẵn thì vẫn kiểm SHA. Đây không phải
runtime preflight. Khi chạy thật, mọi input phải tồn tại và hash phải khớp.
Mỗi lần chạy, kể cả plan-only, dùng tên mới; `--run-name` đã tồn tại bị từ chối.

## Ngân sách phải khai báo trước

`--profile` là nhãn bằng chứng, không âm thầm đổi số bước hay epoch.

| Profile | Cờ minh họa | Diễn giải |
| --- | --- | --- |
| Bootstrap | `--profile bootstrap --static-epochs 1 --updater-epochs 1 --train-max-subjects 2 --static-max-steps 2 --updater-max-steps 2` | Kiểm vận hành và backward; không phải model đã hội tụ |
| PROVISIONAL pilot | `--profile provisional --static-epochs 20 --updater-epochs 20 --train-max-subjects 16` | Ví dụ ngân sách cần review theo tài nguyên và learning curves; chưa được chứng minh đủ cho paper |
| Ngân sách khai báo riêng | Lệnh server ở trên với epoch do người chạy chọn; bỏ `--train-max-subjects` để dùng toàn bộ producer-fit role | Vẫn là PROVISIONAL; không suy ra convergence từ số epoch |

R1 luôn giới hạn hai subject/hai update vì đó là smoke riêng. Giới hạn này
không áp dụng cho R3/R4. Nếu không truyền `--static-max-steps` hoặc
`--updater-max-steps`, không có cap update bổ sung từ runner; config phải có
`stage_options.max_updates=null`. Epoch là lượt qua cohort producer-fit,
không phải số iteration. Muốn kiểm nhiều training seed, chạy lại lệnh với
seed khác và thư mục mới. R4B luôn dùng ba random-control seed 17/29/41;
chúng không phải ba lần huấn luyện độc lập.

## Các stage thực sự được gọi

1. R0: real `preflight`, roles và `runbook-check`.
2. R1: synthetic CPU S0 smoke, rồi real producer-fit S0 smoke; giữ inference,
   resume và runtime receipts riêng.
3. R2 tùy chọn `--benchmark`: checkpoint real R1; hai subject, hai state,
   bốn candidate, Q64, mặc định ba repeat. Đây là parity/resource pilot giới
   hạn; không suy ra speedup toàn pipeline.
4. R3: train B0/B1/B2/B-light bằng cùng ngân sách và seed. Sau đó evaluate
   **checkpoint cuối cùng** của cả bốn arm trên bốn subject validation,
   `scenario=noop`, budget 0, dùng `resolved_config.json` riêng của từng arm.
5. R4: U-only và U+spectral đều bắt đầu từ **cùng bytes checkpoint R3-B2**.
   B2 được khai báo trước; runner không chọn static arm theo training loss
   hoặc chất lượng validation và không tự chọn best epoch.
6. R4B: `headroom-evaluate --exact-pool-audit` riêng cho mỗi updater cuối cùng,
   cùng static base. Runner kiểm bốn validation subject ID và thứ tự khớp
   toàn bộ static evaluations và hai updater audits. Mỗi audit giữ pool32,
   Q1024 screening, exact confirmation và exact retained-pool audit hiện có.
7. CLI `package` gom stage receipts và metadata/log của runner thành một ZIP.

Sau R4B script kết thúc với `r5_status=CLOSED`. Không có bank-build,
ValueNet fitting, calibration, learned STOP, adaptive policy hay lựa chọn
checkpoint bằng target được thêm vào chuỗi. Exit 0 chỉ cho biết chuỗi phần mềm
và archive đã hoàn tất. Headroom bốn subject vẫn **INCONCLUSIVE**, không mở R5.

## Metric và log để gửi lại

Run mới có cấu trúc:

```text
<run-root>/<fresh-name>/
  metadata/
    pipeline_manifest.json
    pipeline_events.jsonl
    pipeline_summary.json
    pipeline_metrics.csv
    stage_logs/<stage>/command.json
    stage_logs/<stage>/stdout.txt
    stage_logs/<stage>/stderr.txt
    stage_logs/<stage>/exit.json
  stages/<stage>/... CLI artifacts, checkpoints, metrics, histories ...
  package/evidence/evidence.zip
  package/manifest.json
  package_logs/... local packaging diagnostics ...
```

Gửi lại `package/evidence/evidence.zip`. Đường dẫn tuyệt đối và SHA256 ZIP
được ghi trong `metadata/pipeline_summary.json`. ZIP chứa snapshot summary
ngay trước khi đóng gói; summary cục bộ sau đó bổ sung kết quả packaging và
hash archive. Log của chính package ở ngoài nguồn đóng gói để snapshot không
thay đổi lúc packager đang hash. Checkpoint `.pt`, target, volume và prediction
tensors không nằm trong ZIP; vẫn giữ trên server để tái chạy.

CSV dùng dạng dài: `stage, arm, subject_id, training_seed, random_seed,
seed_kind, metric_scope, scope, metric, value, artifact, json_path`.
Mỗi row trỏ chính xác tới artifact/field gốc. Summary JSON chứa cùng `rows`
và trạng thái/thời gian/argv/artifact hashes của từng stage.

| Nhóm | Trường đọc từ kết quả có sẵn |
| --- | --- |
| Reconstruction | `before/after/improvement` của MAE, PSNR, SSIM, `masked_charbonnier`; mask/range/denominator và formula/version khi nguồn cung cấp |
| Phạm vi chất lượng | `training_forward_pre_update` cho paired metrics lấy trong forward huấn luyện; `heldout_final_checkpoint` cho validation của checkpoint cuối cùng |
| Seed/subject | Subject ID thật; training seed khác random-control seed; không gộp random seed thành independent subject |
| Exact32 | `top1_regret`, `exact_best_gain`, `observed_best_gain`, pool coverage, pairwise order agreement, sign confusion/fraction, consistency của confirmation |
| Runtime | Host wall time/exit của từng CLI; `diagnostic_timing.phases`; counters/parameter counts/bytes/runtime device và CUDA memory được CLI đo |
| Provenance | Config/checkpoint hash, input hash, runner/CLI SHA, command argv, source/runtime receipts và đường dẫn mọi artifact được giữ |

Không có metric thì để `null` (CSV ghi literal `null`), không điền 0, không
suy diễn FLOPs, không tính average training loss hoặc aggregate mới. Lý do
không đo được, trạng thái, metric policy và version được giữ khi nguồn có.
Full exact32 action/gain/rank rows nằm trong `next1/headroom_metrics.json`
và `next1/next1_evidence.json`; CSV chính chỉ giữ scalar tóm tắt của từng
subject để không nhân bản các mảng action. Counts và phase timing có scope
riêng: không cộng phase con vào tổng enclosing time; CUDA peak là scope CLI
ghi nhận, không tự gọi là peak riêng một kernel.

ZIP giữ nguyên `stage_runtime.json` với schema `pfgr-lite-stage-runtime-v1`,
bao gồm execution config, seed, tên optimizer parameters và sample order.
Trạng thái RNG chỉ là metadata `present`/tên stream; raw RNG state và
checkpoint tensors không được đưa vào ZIP. Full `s0/s1/stage_history.jsonl`
được giữ; CSV/summary chỉ sao chép 100 records cuối mỗi stage và ghi rõ scope.

`stage_history.jsonl` đầy đủ được giữ trong ZIP. Summary/CSV chỉ chép tối đa
100 record cuối mỗi training stage, scope `training_history_last_100_records`,
với epoch/update/loss/gradient thực và số dòng gốc; không lấy trung bình
history và không gọi phần tail này là toàn bộ learning curve.

## Khi lỗi

Runner dừng ngay stage phụ thuộc sau lỗi, giữ stdout/stderr/traceback và
attempted exit code, rồi vẫn thử CLI package trên phần evidence hiện có.
Missing output, changed checkpoint/config bytes, sai cohort và lỗi collect
metric đều làm run failed. Timeout tùy chọn bằng `--stage-timeout-seconds`;
runner terminate process group, giữ mã 124, rồi thử package. Keyboard
interrupt khi chờ child giữ mã 130 và cũng thử package. Không thể đảm bảo
đóng gói sau SIGKILL, hết disk hoặc process/host chết; các log đã flush vẫn ở
run directory.

Không tự resume và không skip bằng marker. Muốn chạy lại dùng thư mục mới;
checkpoint/resume artifact đã tạo vẫn được giữ cho workflow CLI riêng được
review. Không thay đổi source trong lúc chuỗi đang chạy.

Packaging mặc định runner cho phép mỗi file tối đa 64 MiB và tổng archive
512 MiB; đổi bằng `--package-max-file-mib`/`--package-max-archive-mib` khi
budget log lớn hơn. Guard nội dung của CLI vẫn áp dụng. Kiểm
`package/manifest.json` để thấy file bị loại và lý do; trạng thái
`MISSING_REQUIRED` của bộ metadata không đồng nghĩa có bằng chứng khoa học.
Nếu package thất bại, gửi `metadata/pipeline_summary.json` và log lỗi package
trước, đồng thời giữ nguyên run để khắc phục.

## Kiểm chứng phần mềm

Test runner dùng mocked compute cùng file/argv thật và packager thật, không
huấn luyện hoặc đọc BraTS. Các case bao phủ checkpoint/config dependency,
hai updater dùng cùng B2, final validation cohort, parser hiện tại, quoting,
hash mismatch, no-overwrite, child nonzero/timeout, packaging khi lỗi,
metric/null/seed scope và không có R5. Frontend CPU smoke vẫn là gate riêng.
Software PASS không chứng minh runtime CUDA, pretrained origin, hội tụ,
chất lượng tái tạo hay clinical utility.
