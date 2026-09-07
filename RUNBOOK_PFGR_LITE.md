# PFGR-Lite R0--R10: runbook tiếng Việt, có thể chạy và có điểm dừng

Runbook này là lối vào vận hành phần mềm PFGR-Lite. Nó không phải bằng chứng
BraTS21, GPU/CUDA, checkpoint đã huấn luyện, chất lượng tái tạo hay tuyên bố
lâm sàng. Tất cả lệnh dưới đây gọi đúng `python -m smagm.cli.pfgr_lite`; các
đường dẫn checkpoint, split, roles, hash và cohort phải lấy từ receipt thật,
không được đoán theo tên thư mục.

## Trạng thái bằng chứng đã biết

Root đã chạy toàn repository trong checkout sạch tại `abe252d`: **982 passed,
18 skipped, 26 subtests, 1 warning trong 115.34 s**. Sau commit packaging
`3f08288`, kiểm tra artifact/runbook/E2E trong checkout sạch đạt **35 passed
trong 33.60 s**; các scope chồng lấp, không cộng thành một tổng mới. Phép đóng
gói nhiều run độc lập giữ đúng 38 file metadata, gồm split/roles với 1.251 ID
tự sinh, và loại checkpoint/bank tensor. Ba automated checks của
`POINT_GUIDED_FRONTEND` đều **PASS**; Human Gate vẫn pending.

Đây là software evidence, không phải bằng chứng khoa học. CUDA/AMP,
MedicalNet pretrained thật, dữ liệu bệnh nhân, checkpoint huấn luyện, latency
ổn định, speedup, headroom và chất lượng tái tạo vẫn **PENDING**. Historical
smoke SHA cũ chưa được xác minh. Các run Reward/trajectory cũ chỉ là tài liệu
lịch sử; dùng runbook này làm entrypoint PFGR-Lite.

## 0. Quy tắc an toàn, thứ tự môi trường và biến chung

Trước hết bảo toàn thay đổi địa phương; không `reset`, `clean`, checkout hoặc
ghi đè run cũ. Nếu cần checkout sạch, người vận hành tạo bản sao/worktree
được phê duyệt; runbook không tự xoá thay đổi. Sau đó tạo/kiểm tra venv trong
repository, cài extras vào **repo venv** (hoặc chỉ xác thực interpreter đã có
nếu `POINT_GUIDED_PYTHON` trỏ ra ngoài), rồi mới kiểm tra sáu biến đầu vào. Không
giả định A4000, hai GPU, CUDA hay đường dẫn máy chủ cụ thể; `PFGR_DEVICE` là
giá trị thực do người vận hành chọn.

```bash
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain)" ]; then
  echo "working tree đang dirty; bảo toàn thay đổi và dừng trước khi chạy cohort" >&2
  exit 3
fi
if [ -n "${POINT_GUIDED_PYTHON:-}" ]; then
  test -x "$POINT_GUIDED_PYTHON" || { echo "POINT_GUIDED_PYTHON không executable: $POINT_GUIDED_PYTHON" >&2; exit 2; }
  "$POINT_GUIDED_PYTHON" -c 'import torch, nibabel' || { echo "interpreter ngoài repo thiếu torch/nibabel; cài trong repo venv hoặc sửa POINT_GUIDED_PYTHON" >&2; exit 2; }
else
  if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
    python3 -m venv "$REPO_ROOT/.venv"
  fi
  POINT_GUIDED_PYTHON="$REPO_ROOT/.venv/bin/python"
  "$POINT_GUIDED_PYTHON" -m pip install -e '.[test,real-data,wandb]'
fi

: "${BRATS21_ROOT:?đặt thư mục BraTS21 thật}"
: "${MEDICALNET_CKPT:?đặt file MedicalNet checkpoint cục bộ}"
: "${MEDICALNET_SHA256:?đặt SHA256 mong đợi của checkpoint}"
: "${BASELINE_SPLIT:?đặt file split đã review}"
: "${OUTPUT_ROOT:?đặt thư mục output writable; không bắt buộc nằm trong repo}"
mkdir -p "$OUTPUT_ROOT"
test -d "$OUTPUT_ROOT"
export PYTHONPATH="$REPO_ROOT/src"
export PFGR_DEVICE="${PFGR_DEVICE:-cpu}"
export PFGR_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export WANDB_ENTITY="${WANDB_ENTITY:-khanhlq-work-hanoi-university-of-science-and-technology}"
export WANDB_PROJECT="${WANDB_PROJECT:-smagm-point-guided}"

PFGR_COMMON=(
  --config "$REPO_ROOT/configs/pfgr_lite/main.json"
  --data-root "$BRATS21_ROOT"
  --split-file "$BASELINE_SPLIT"
  --output-root "$OUTPUT_ROOT"
  --device "$PFGR_DEVICE"
  --no-amp
)
PFGR_FRESH_COMMON=(
  "${PFGR_COMMON[@]}"
  --medicalnet-checkpoint "$MEDICALNET_CKPT"
  --medicalnet-sha256 "$MEDICALNET_SHA256"
)

PFGR_R0_DIR="$OUTPUT_ROOT/R0-$PFGR_RUN_ID"
PFGR_ROLES="$PFGR_R0_DIR/roles.json"
PFGR_R1_SYNTH_DIR="$OUTPUT_ROOT/R1-synthetic-$PFGR_RUN_ID"
PFGR_R1_REAL_DIR="$OUTPUT_ROOT/R1-real-$PFGR_RUN_ID"
PFGR_STATIC_DIR="$OUTPUT_ROOT/R3-b2-$PFGR_RUN_ID"
PFGR_STATIC_CHECKPOINT="$PFGR_STATIC_DIR/inference.pt"
PFGR_R4_DIR="$OUTPUT_ROOT/R4-updater-$PFGR_RUN_ID"
PFGR_BASE_CHECKPOINT="$PFGR_R4_DIR/inference.pt"
PFGR_U_ONLY_CHECKPOINT="$OUTPUT_ROOT/R4-u-only-$PFGR_RUN_ID/inference.pt"
PFGR_R5_DIR="$OUTPUT_ROOT/R5-bank-$PFGR_RUN_ID"
PFGR_BANK_INDEX="$PFGR_R5_DIR/s2/bank/index.json"
PFGR_R6_DIR="$OUTPUT_ROOT/R6-v366-$PFGR_RUN_ID"
PFGR_VALUE_CHECKPOINT="$PFGR_R6_DIR/value.pt"
PFGR_CALIBRATION_REVIEW="$OUTPUT_ROOT/review/calibration-$PFGR_RUN_ID.json"
PFGR_CALIBRATED_CHECKPOINT="$OUTPUT_ROOT/R7-$PFGR_RUN_ID/adaptive.pt"
PFGR_FINAL_REVIEW="$OUTPUT_ROOT/review/final-$PFGR_RUN_ID.json"
PFGR_REVIEWED_RUN_DIR="$OUTPUT_ROOT/R9-reviewed-$PFGR_RUN_ID"
PFGR_VALUE_RESUME="$OUTPUT_ROOT/R6-v366-$PFGR_RUN_ID/value-resume.pt"
# Giữ đúng envelope fit/resume của V: CLI mặc định batch=32 nếu không override.
PFGR_VALUE_EPOCHS="${PFGR_VALUE_EPOCHS:-2}"
PFGR_VALUE_BATCH_SIZE="${PFGR_VALUE_BATCH_SIZE:-32}"

require_artifact() {
  test -e "$1" || { echo "thiếu predecessor artifact: $1" >&2; exit 2; }
}

require_dir() {
  test -d "$1" || { echo "thiếu thư mục đầu vào $2: $1; kiểm tra biến môi trường/cohort" >&2; exit 2; }
}

require_file() {
  test -f "$1" || { echo "thiếu file đầu vào $2: $1; không đổi tên để né kiểm tra" >&2; exit 2; }
}

require_dir "$BRATS21_ROOT" BRATS21_ROOT
require_file "$MEDICALNET_CKPT" MEDICALNET_CKPT
require_file "$BASELINE_SPLIT" BASELINE_SPLIT
```

Khi cần một bản main sạch để phát hành, thao tác trong worktree/copy riêng
(không dùng worktree tích hợp đang có thay đổi). Lệnh đồng bộ là fetch rồi
fast-forward-only; không reset/clean và không tự ghi đè dirty tree:

```bash
set -euo pipefail
test -z "$(git status --porcelain)" || { echo "worktree dirty; giữ nguyên và dừng" >&2; exit 3; }
git fetch origin main
git switch main
git pull --ff-only origin main
git status --short --branch
```

Nếu `git fetch` không truy cập được remote, hoặc `git pull --ff-only` báo
divergence, giữ nguyên tree và ghi lỗi/branch/SHA; không dùng `reset --hard`,
`clean -fd`, force-pull hay chuyển sang branch tích hợp.

Checkpoint MedicalNet lưu ngoài repo phải được người vận hành cung cấp qua
`MEDICALNET_CKPT` và kiểm bằng preflight. Với input thật, `weights.json`/receipt
chỉ ghi các trường CLI thực sự phát hành: `checkpoint`, `sha256`,
`expected_sha256`, `source_input_channels`, `adapted_input_channels`,
`input_conv_adapted`, `checkpoint_integrity_verified`,
`official_pretrained_verified`, `source_state_dict_key_count`,
`loaded_backbone_key_count`, `synthetic_untrained`; không suy diễn từ tên file,
không copy checkpoint vào repo và không dùng artifact field không tồn tại để
thay thế provenance.

**Cổng nguồn runtime hiện tại (phải ghi rõ khi chạy).** Registry nguồn chính
thức đang trống; vì vậy SHA và integrity của file cục bộ chỉ chứng minh đúng
bytes đã nhận, không chứng minh đây là MedicalNet pretrained chính thức. R0/R1
synthetic và mọi kiểm thử phần mềm vẫn có thể PASS với `official_pretrained_verified=false`,
nhưng không được gắn nhãn pretrained, trained producer hay science. Nhánh MAIN
R5 (bank thật) và R7 (calibration thật) bị **BLOCKED** cho tới khi người review
đã vet origin checkpoint, đăng ký đúng digest qua thay đổi repository được duyệt,
và receipt ghi đủ provenance; runbook không tự đăng ký, không tự nới guard, và
không coi một đường dẫn/hashes đoán được là nguồn đã duyệt. Nếu chỉ cần kiểm
tra engineering an toàn, dùng `configs/pfgr_lite/synthetic.json` cho R0/R1 và
đánh dấu rõ rằng đó là generated input, không phải real-input smoke.

Mỗi block shell phải qua `bash -n` trước khi chạy. `--dry-manifest` chỉ kiểm
tra parser/phụ thuộc dự kiến, **không** phải execution evidence. Mỗi lệnh ghi
`argv`, exit code, source SHA/dirty diff, config/effective-policy hash,
device/precision, input/output hashes, role/capability, counters và traceback
đầu tiên (nếu lỗi). W&B chỉ ghi URL/run ID do W&B trả về; không tự chế URL.

### Review receipt do người thật ký

Review receipt không được CLI tự phê duyệt. Reviewer tạo file sau khi đã xem
receipt thật, dùng schema `pfgr-lite-review-receipt-v1` gồm `scope`,
`decision`, `reviewer`, `created_at`, `config_hash`, `cohort_hash`, và các
digest trong mapping `artifacts` của `expected_artifacts` từ
`review_context.json` (`checkpoint_sha256`, `value_checkpoint_sha256` nếu có,
`role_manifest_digest`, `split_hash`). Không dùng chuỗi `not_applicable`, hash giả,
placeholder hay cohort tự bịa. `decision=ENGINEERING_DIAGNOSTIC` chỉ dành cho
fixture synthetic; production cần `APPROVED`. `--review-receipt` được truyền
cho R7/R9; `calibrate --evidence` chỉ là import diagnostic synthetic, không
thay thế collection S5 thật.

`cohort_hash` phải là hash của payload `pfgr-lite-review-context-v1` do CLI
tạo trước khi chạy target/route: `selected_subject_ids`, `split_role`,
`baseline_split_hash`, `role_manifest_digest`, `producer_compatibility_hash`,
`value_fit_identity_hash` (nếu dùng V), `config_hash`, `policy`, `budget`,
`seed`, `teacher_mode`, `query_count`, `candidate_count`,
`candidate_chunk_size`, `decode_chunk_size`. `artifacts` phải chứa hash thật
`checkpoint_sha256`, `value_checkpoint_sha256` (nếu có), `role_manifest_digest`
và `split_hash`; reviewer không tự tính cohort khác. Tạo request bằng
`--dry-manifest` của đúng `calibrate` (R7) hoặc `evaluate --split-role test`
(R9), đọc `review_context.json`, rồi ghi receipt mới vào thư mục riêng trước
khi chạy lệnh thật.

Reviewer phải dùng writer dưới đây sau khi đọc **toàn bộ**
`review_context.json`; script không có giá trị mặc định cho reviewer/decision,
không tự APPROVED và không ghi đè file đã tồn tại. Nó chép
`expected_artifacts` của context vào trường `artifacts` mà CLI validator yêu
cầu, đồng thời khóa nguyên `schema_version`, `scope`, `config_hash` và
`cohort_hash`; production chỉ dùng `PFGR_REVIEW_DECISION=APPROVED`, còn
fixture synthetic chỉ dùng `ENGINEERING_DIAGNOSTIC`.

```bash
: "${PFGR_REVIEW_CONTEXT:?đặt path review_context.json từ dry-manifest}"
: "${PFGR_REVIEW_RECEIPT:?đặt path receipt output mới, không được tồn tại}"
: "${PFGR_REVIEWER:?reviewer phải nhập danh tính thật, không dùng placeholder}"
: "${PFGR_REVIEW_DECISION:?nhập APPROVED hoặc ENGINEERING_DIAGNOSTIC, không để trống}"
"$POINT_GUIDED_PYTHON" - "$PFGR_REVIEW_CONTEXT" "$PFGR_REVIEW_RECEIPT" \
  "$PFGR_REVIEWER" "$PFGR_REVIEW_DECISION" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

context_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
reviewer = sys.argv[3].strip()
decision = sys.argv[4].strip()
if not reviewer or decision not in {"APPROVED", "ENGINEERING_DIAGNOSTIC"}:
    raise SystemExit("reviewer/decision phải là input tường minh; decision không hợp lệ")
if not context_path.is_file():
    raise SystemExit(f"thiếu review_context.json: {context_path}")
context = json.loads(context_path.read_text(encoding="utf-8"))
required = {"schema_version", "status", "scope", "context", "cohort_hash", "config_hash", "expected_artifacts", "decision_required", "scientific_status"}
if set(context) != required:
    raise SystemExit(f"review_context keys sai; cần đúng {sorted(required)}")
if context["schema_version"] != "pfgr-lite-review-context-v1" or context["status"] != "REVIEW_REQUIRED" or context["scientific_status"] != "NOT_EVALUATED":
    raise SystemExit("context không phải review request v1 chưa được quyết định")
if context["decision_required"] is not True:
    raise SystemExit("context không yêu cầu reviewer quyết định")
for name in ("scope", "cohort_hash", "config_hash"):
    if not isinstance(context[name], str) or not context[name].strip():
        raise SystemExit(f"context thiếu {name}")
expected = context["expected_artifacts"]
if not isinstance(expected, dict) or not expected or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in expected.items()):
    raise SystemExit("expected_artifacts phải là mapping path/hash thật")
if output_path.exists():
    raise SystemExit(f"refuse overwrite reviewer receipt: {output_path}")
if not output_path.parent.is_dir():
    raise SystemExit(f"receipt parent phải được tạo riêng trước: {output_path.parent}")
payload = {
    "schema_version": "pfgr-lite-review-receipt-v1",
    "scope": context["scope"],
    "decision": decision,
    "reviewer": reviewer,
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "config_hash": context["config_hash"],
    "cohort_hash": context["cohort_hash"],
    "artifacts": expected,
}
with output_path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
    handle.write("\n")
print(json.dumps({"receipt": str(output_path), "scope": payload["scope"], "decision": decision}, sort_keys=True))
PY
```

## R0 -- preflight, provenance và test phần mềm

**Mục đích.** Kiểm tra interpreter, dependencies, device thật, SHA checkpoint,
adaptation MedicalNet, split gốc, normalization và roles mà chưa đọc target
held-out. **Tiền đề.** Sáu biến môi trường đã kiểm tra; working tree sạch theo
block trên. **Lệnh chính:**

```bash
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite preflight "${PFGR_FRESH_COMMON[@]}" \
  --write-roles --run-name "R0-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m pytest -q tests/features/point_guided/pfgr_lite/test_runbook.py
"$POINT_GUIDED_PYTHON" -m pytest -q tests/features/point_guided/pfgr_lite
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite runbook-check \
  --runbook "$REPO_ROOT/RUNBOOK_PFGR_LITE.md" \
  --config-dir "$REPO_ROOT/configs/pfgr_lite" \
  --output-root "$OUTPUT_ROOT" --run-name "R0-runbook-check-$PFGR_RUN_ID"
```

Nhánh synthetic bounded (không cần sáu biến real) chỉ để kiểm parser và
preflight:

```bash
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite preflight --synthetic \
  --config "$REPO_ROOT/configs/pfgr_lite/synthetic.json" \
  --output-root "$OUTPUT_ROOT" --run-name "R0-synthetic-$PFGR_RUN_ID" \
  --dry-manifest
```

**Files/metrics.** `environment.json`, `source.json`, `weights.json`,
`split.json`, `roles.json`, `resolved_config.json`, `receipt.json`; kiểm
`target_reads=0`, SHA thực tế/expected, `input_conv_adapted`,
`checkpoint_integrity_verified`, `official_pretrained_verified`, split hash và
role digest. SHA match và `checkpoint_integrity_verified=true` chỉ chứng minh
đúng bytes local; `official_pretrained_verified=false` vẫn là trạng thái hợp lệ
nếu registry/chứng cứ chính thức chưa có, nhưng không được gọi pretrained hay
chất lượng thật. **SOFTWARE_PASS** nếu mọi identity hợp lệ; **SOFTWARE_FAIL** nếu
SHA/config/device/schema sai; **INCONCLUSIVE/BLOCKED** nếu thiếu real input,
không được tự resplit. **Tiếp:** chỉ đi R1 khi receipt PASS; mismatch thì
STOP, sửa đúng input rồi lặp R0.

## R1 -- smoke S0 bounded và checkpoint thật

**Mục đích.** Chạy stage service thật trên tối đa 2 subject, tối đa 2 bước;
đây không phải S1/U hay checkpoint đã huấn luyện. **Tiền đề.** R0 PASS hoặc
nhánh synthetic được gắn engineering-only. Lệnh synthetic này thực sự chạy
stage, không phải manifest:

```bash
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite smoke --synthetic \
  --config "$REPO_ROOT/configs/pfgr_lite/synthetic.json" \
  --output-root "$OUTPUT_ROOT" --run-name "R1-synthetic-$PFGR_RUN_ID" \
  --device cpu --no-amp --max-subjects 2 --max-steps 2
require_artifact "$PFGR_R1_SYNTH_DIR/inference.pt"
require_artifact "$PFGR_R1_SYNTH_DIR/resume.pt"
```

Nhánh real train-only (không đọc held-out target) dùng đúng checkpoint
MedicalNet và roles R0:

```bash
require_artifact "$PFGR_ROLES"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite smoke "${PFGR_FRESH_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --run-name "R1-real-$PFGR_RUN_ID" \
  --max-subjects 2 --max-steps 2
require_artifact "$PFGR_R1_REAL_DIR/inference.pt"
require_artifact "$PFGR_R1_REAL_DIR/resume.pt"
```

**Files/metrics.** stage receipt, `inference.pt`, `resume.pt`,
`stage_state.json`, `stage_runtime.json`, gradient/frozen-BN/projector hashes,
MedicalNet traversal count, observation/target reads. **SOFTWARE_PASS** chỉ
xác nhận forward/backward finite, graph và target boundary; khoa học là
**NOT_EVALUATED/INCONCLUSIVE**. **FAIL** nếu nonfinite, frozen gradient sai,
target leak, missing runtime; **Tiếp:** R2/R3 sau review pilot; không suy ra
chất lượng từ synthetic/untrained.

## R2 -- parity benchmark cùng work và resource pilot

**Mục đích.** Dùng cùng state, action delta đã lưu, voxel IDs, dtype, chunk và
MLP để so full-write reference với sparse query delta; đo cold/warm cache sau
khi parity. **Tiền đề.** R1 phải có checkpoint `inference.pt`; ưu tiên producer
thật R1, còn synthetic chỉ là engineering pilot. R4 checkpoint chỉ là một lần
rerun parity tùy chọn sau khi updater đã train, không phải predecessor bắt buộc.
Chọn predecessor thật bằng lệnh:

```bash
if [ -f "$PFGR_R1_REAL_DIR/inference.pt" ]; then
  PFGR_R2_CHECKPOINT="$PFGR_R1_REAL_DIR/inference.pt"
  PFGR_R2_CAPABILITY="production_pending"
  PFGR_R2_ARGS=("${PFGR_COMMON[@]}" --roles-file "$PFGR_ROLES")
elif [ -f "$PFGR_R1_SYNTH_DIR/inference.pt" ]; then
  PFGR_R2_CHECKPOINT="$PFGR_R1_SYNTH_DIR/inference.pt"
  PFGR_R2_CAPABILITY="engineering_only"
  PFGR_R2_ARGS=(
    --synthetic --config "$REPO_ROOT/configs/pfgr_lite/synthetic.json"
    --output-root "$OUTPUT_ROOT" --device cpu --no-amp
  )
else
  echo "thiếu R1 inference.pt; chạy R1 trước khi benchmark" >&2
  exit 2
fi
require_artifact "$PFGR_R2_CHECKPOINT"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite benchmark "${PFGR_R2_ARGS[@]}" \
  --checkpoint "$PFGR_R2_CHECKPOINT" \
  --run-name "R2-$PFGR_RUN_ID" --max-subjects 2 --max-states 2 \
  --candidate-count 4 --teacher-mode iid_fixed_q --query-count 64 --repeats 3
```

Nếu rơi vào nhánh synthetic thì phải giữ nguyên `--synthetic` và
`synthetic.json`; không trộn checkpoint synthetic với `main.json`, data-root,
roles hoặc split thật. Nhánh real mới được dùng `PFGR_COMMON` và `PFGR_ROLES`;
`PFGR_R2_CAPABILITY` phải đi cùng receipt để không biến engineering pilot thành
producer production.

Nếu muốn tách parity của producer R4, lặp lại đúng lệnh sau khi R4 có
`PFGR_BASE_CHECKPOINT`, giữ `PFGR_R2_CAPABILITY` và provenance riêng; không đổi
tên checkpoint để giả vờ cùng producer. Synthetic/R1 reduced pilot luôn ghi
`scientific_status=NOT_EVALUATED`.

**Files/metrics bắt buộc.** `benchmark.json`, `rows.jsonl`, `parity.json`,
`service_receipt.json`, `receipt.json`; actual states/actions/candidates,
Q/draws/unique voxels, valid sphere voxels và padded slots, footprint union,
query/MLP/decoder calls, full-plane clone bytes, allocated/reserved CUDA
memory (hoặc null CPU), device list, cache build/hit/validation time, cold và
warm elapsed. Tách **same-work** parity khỏi **less-work** Q/state reduction;
không gọi speedup nếu repeats/noise/chunks khác. FP64 tolerance
`1e-10/1e-9`, FP32 `1e-6/1e-5`; near-boundary mismatch là unresolved boundary.
**PASS** là parity trong ngưỡng và counters khớp; **FAIL** là mismatch/
missing saved work; **INCONCLUSIVE** là reduced work, noisy timing hoặc CPU
không có memory GPU. **Tiếp:** R3/R4; parity FAIL thì STOP benchmark và sửa
writer/lattice.

## R3 -- static B0/B1/B2/B-light control

**Mục đích.** So bốn static heads với cùng split/work, source distinguishability,
full-affine geometry và D(Z0); không claim convergence. **Tiền đề.** R0/R1
PASS, roles thực. Lệnh tách run và không tự mở rộng:

```bash
require_artifact "$PFGR_ROLES"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite static-train "${PFGR_FRESH_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --base b2 --epochs 1 --max-subjects 2 \
  --run-name "R3-b2-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite static-train "${PFGR_FRESH_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --base b1 --epochs 1 --max-subjects 2 \
  --run-name "R3-b1-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite static-train "${PFGR_FRESH_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --base b0 --epochs 1 --max-subjects 2 \
  --run-name "R3-b0-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite static-train "${PFGR_FRESH_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --base b_light --epochs 1 --max-subjects 2 \
  --run-name "R3-blight-$PFGR_RUN_ID"
require_artifact "$PFGR_STATIC_CHECKPOINT"
```

One epoch là engineering smoke, không phải convergence. Record parameter
counts, source hashes, D(Z0) MAE/PSNR/SSIM/Charbonnier với mask, denominator và
data range cố định. **PASS** là variant/shape/hash đúng; **FAIL** nếu
capacity/affine/modality hoặc frozen ownership sai; scientific
**INCONCLUSIVE** cho pilot. **Tiếp:** chọn static producer bằng review rồi R4;
static-only không cung cấp ValueBank producer MAIN.

## R4 -- updater, correction headroom và Oracle scope

**Mục đích.** Train hai arm có provenance riêng: `u_plus_spectral` MAIN và
`u_only` control. Mọi correction/headroom/random/oracle control chính dùng
**cùng checkpoint U+spectral đã train** (`PFGR_BASE_CHECKPOINT`), không dùng
static cold U; U-only được báo riêng. Không claim spectral utility nếu
projector random/frozen chưa có gradient/update evidence. **Tiền đề.** R3
static checkpoint đã chọn.

```bash
require_artifact "$PFGR_STATIC_CHECKPOINT"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite updater-train "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_STATIC_CHECKPOINT" \
  --spectral-arm u_plus_spectral --epochs 1 --max-subjects 2 \
  --run-name "R4-updater-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite updater-train "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_STATIC_CHECKPOINT" \
  --spectral-arm u_only --epochs 1 --max-subjects 2 \
  --run-name "R4-u-only-$PFGR_RUN_ID"
require_artifact "$PFGR_BASE_CHECKPOINT"
require_artifact "$PFGR_U_ONLY_CHECKPOINT"
```

Các controls sau đây là lệnh độc lập, không phải matrix ngầm; mọi control
chính dùng `PFGR_BASE_CHECKPOINT`:

```bash
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --scenario static --budget 0 --max-subjects 2 --split-role validation \
  --run-name "R4-static-k0-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --scenario noop --budget 0 --max-subjects 2 --split-role validation \
  --run-name "R4-noop-k0-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --scenario random --budget 1 --max-subjects 2 --split-role validation \
  --local-footprint-audit --run-name "R4-random-k1-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --scenario random --budget 2 --max-subjects 2 --split-role validation \
  --local-footprint-audit --run-name "R4-random-k2-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --scenario random --budget 4 --max-subjects 2 --split-role validation \
  --local-footprint-audit --run-name "R4-random-k4-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_U_ONLY_CHECKPOINT" \
  --scenario random --budget 2 --max-subjects 2 --split-role validation \
  --run-name "R4-u-only-random-k2-$PFGR_RUN_ID"
```

Oracle phải giữ candidate generation target-free trước target join. `sampled_one`
và `greedy` iid luôn có confirmation độc lập; `greedy` không phải optimum toàn
cục. Tiny exact subset và all-N được tách rõ ở dưới:

```bash
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite oracle-evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --oracle-mode sampled_one --budget 1 --candidate-count 32 \
  --teacher-mode iid_fixed_q --query-count 1024 \
  --confirmation-mode iid_fixed_q --confirmation-query-count 1024 \
  --max-subjects 2 --split-role validation --run-name "R4-oracle-sampled1-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite oracle-evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --oracle-mode greedy --budget 2 --candidate-count 32 \
  --teacher-mode iid_fixed_q --query-count 1024 \
  --confirmation-mode iid_fixed_q --confirmation-query-count 1024 \
  --max-subjects 2 --split-role validation --run-name "R4-oracle-greedy-k2-$PFGR_RUN_ID"
```

Tiny exact subset phải dùng `sampled_one`, không được đặt tên `all_exact_one`:

```bash
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite oracle-evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --oracle-mode sampled_one --budget 1 --candidate-count 8 \
  --teacher-mode exact_footprint --confirmation-mode exact_footprint \
  --max-subjects 2 --split-role validation --run-name "R4-oracle-exact-subset8-$PFGR_RUN_ID"
```

Lệnh trên lấy tối đa tám candidate từ bank đã niêm phong rồi đo exact
footprint; `confirmation_mode=exact_footprint` là phép đo độc lập cho winner.
`all_exact_one` **không cắt theo `candidate_count`** trong implementation hiện
tại: nó đo toàn bộ candidate hợp lệ. Vì vậy all-N exact là pilot đắt tiền,
riêng và cần review:

```bash
if [ "${PFGR_RUN_ALL_N_ORACLE:-0}" = 1 ]; then
  "$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite oracle-evaluate "${PFGR_COMMON[@]}" \
    --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
    --oracle-mode all_exact_one --budget 1 \
    --teacher-mode exact_footprint --max-subjects 1 --split-role validation \
    --run-name "R4-oracle-all-n-reviewed-$PFGR_RUN_ID"
else
  echo "bỏ qua all-N Oracle; cần review và PFGR_RUN_ALL_N_ORACLE=1"
fi
```

**So sánh paired callable.** Sau khi có random K2 và Oracle greedy K2, dùng
đúng `compare_paired_artifacts`, không bịa gain, và ghi artifact mới:

```bash
PFGR_COMPARE_DIR="$OUTPUT_ROOT/R4-paired-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" - \
  "$OUTPUT_ROOT/R4-random-k2-$PFGR_RUN_ID/metrics.json" \
  "$OUTPUT_ROOT/R4-random-k2-$PFGR_RUN_ID/paired_subjects.jsonl" \
  "$OUTPUT_ROOT/R4-oracle-greedy-k2-$PFGR_RUN_ID/privileged_oracle.jsonl" \
  "$OUTPUT_ROOT/R4-oracle-greedy-k2-$PFGR_RUN_ID/receipt.json" \
  "$PFGR_COMPARE_DIR/r4-paired.json" <<'PY'
import json
import sys
from pathlib import Path
from smagm.features.point_guided.pfgr_lite.metrics import ComparisonOptions, compare_paired_artifacts

metrics_path, paired_path, oracle_path, oracle_receipt_path, output_path = map(Path, sys.argv[1:])
output_path.parent.mkdir(parents=True, exist_ok=False)
if output_path.exists():
    raise FileExistsError(f"comparison output already exists: {output_path}")
oracle_receipt = json.loads(oracle_receipt_path.read_text(encoding="utf-8"))
oracle_meta = oracle_receipt.get("metrics", {}).get("source_receipt")
if not isinstance(oracle_meta, dict):
    raise ValueError("oracle receipt has no aggregate source_receipt")
result = compare_paired_artifacts(
    None,
    {"metrics_path": metrics_path, "paired_subjects_path": paired_path},
    {"output_path": oracle_path, "source_receipt": oracle_meta},
    # Keep the declared default (32 independent subjects).  A two-subject
    # pilot is useful for software joins only and must remain INCONCLUSIVE.
    options=ComparisonOptions(),
)
output_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"software_status": result["software_status"], "scientific_status": result["scientific_status"], "r4": result["r4_decision"]}, sort_keys=True))
PY
```

**Metrics/branches.** Receipts phải ghi signed gain, benefit/harm/neutral
denominator, M/mask, Q/SE/scope, candidate coverage, z0/final metrics, U và
spectral gradient/frozen hashes, state/action identity, `R(Z0)-R(ZK)` và
telescoping residual. **SOFTWARE_PASS** nếu route/teacher/identities đúng;
**FAIL** nếu target leak, mismatch, stale action, fabricated all-N hoặc thiếu
confirmation; scientific **PASS** chỉ khi margin/paired uncertainty đạt,
**FAIL** khi đủ power chứng minh không material, thiếu power/noisy là
**INCONCLUSIVE**. Nếu oracle-Z0 không material thì STOP router và review U/base;
random≈oracle cùng dương chỉ hỗ trợ correction, không hỗ trợ learned selection;
oracle>random nhưng learned kém thì kiểm bank/V/ranking; learned hữu ích mới
được review calibration.

## R5 -- bank immutable và replay audit

**Mục đích.** Hoàn tất forced target-free traces rồi mới đo labels; candidate
subset có scope rõ, exact/fixed-Q version, shard/index/replay bất biến.
**Tiền đề.** R4 **U+spectral trained** `PFGR_BASE_CHECKPOINT`, roles R0.

```bash
require_artifact "$PFGR_BASE_CHECKPOINT"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite bank-build "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --teacher-mode iid_fixed_q --query-count 1024 --candidate-count 32 \
  --max-states 3 --max-subjects 2 --run-name "R5-bank-$PFGR_RUN_ID"
require_artifact "$PFGR_BANK_INDEX"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite bank-verify "${PFGR_COMMON[@]}" \
  --bank-index "$PFGR_BANK_INDEX" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --split-file "$BASELINE_SPLIT" --roles-file "$PFGR_ROLES" \
  --replay-count 2 --run-name "R5-verify-$PFGR_RUN_ID"
```

`index.json`, shard hashes, role/split/producer/writer/query/lattice hashes,
gain scale và replay outcomes phải khớp. Q1024 là pilot, không phải bằng chứng
label precision. **SOFTWARE_PASS** cần index/shard/replay identity thật;
**FAIL** stale/mixed role/missing scale; scientific **INCONCLUSIVE** nếu Q hoặc
cohort nhỏ. S2 không có optimizer resume: nếu bank-build bị gián đoạn, giữ
thư mục partial để điều tra, không append/reuse; chạy lại bằng run-name mới và
xác minh shard atomic/không overwrite.

## R6 -- V same-bank 126/270/366 và V222 tùy chọn

**Mục đích.** Fit/evaluate các descriptor trên **cùng bank rows/order/scale**,
split/roles/producer; fit có zero teacher/U/D/target reads. **Tiền đề.** R5
verify PASS và producer checkpoint R4. V366 là MAIN; V126/V270 controls; V222
chỉ chạy khi `PFGR_RUN_V222=1`, không tự mở rộng matrix.

```bash
require_artifact "$PFGR_BANK_INDEX"
require_artifact "$PFGR_BASE_CHECKPOINT"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite value-fit \
  --config "$REPO_ROOT/configs/pfgr_lite/main.json" \
  --data-root "$BRATS21_ROOT" \
  --bank-index "$PFGR_BANK_INDEX" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --split-file "$BASELINE_SPLIT" --roles-file "$PFGR_ROLES" \
  --value-input 366 --epochs "$PFGR_VALUE_EPOCHS" --batch-size "$PFGR_VALUE_BATCH_SIZE" --output-root "$OUTPUT_ROOT" \
  --run-name "R6-v366-$PFGR_RUN_ID" --device "$PFGR_DEVICE" --no-amp
for value_variant in 126 270; do
  "$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite value-fit \
    --config "$REPO_ROOT/configs/pfgr_lite/main.json" \
    --data-root "$BRATS21_ROOT" \
    --bank-index "$PFGR_BANK_INDEX" --checkpoint "$PFGR_BASE_CHECKPOINT" \
    --split-file "$BASELINE_SPLIT" --roles-file "$PFGR_ROLES" \
    --value-input "$value_variant" --epochs "$PFGR_VALUE_EPOCHS" --batch-size "$PFGR_VALUE_BATCH_SIZE" --output-root "$OUTPUT_ROOT" \
    --run-name "R6-v${value_variant}-$PFGR_RUN_ID" --device "$PFGR_DEVICE" --no-amp
done
if [ "${PFGR_RUN_V222:-0}" = 1 ]; then
  "$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite value-fit \
    --config "$REPO_ROOT/configs/pfgr_lite/main.json" \
    --data-root "$BRATS21_ROOT" \
    --bank-index "$PFGR_BANK_INDEX" --checkpoint "$PFGR_BASE_CHECKPOINT" \
    --split-file "$BASELINE_SPLIT" --roles-file "$PFGR_ROLES" \
    --value-input 222 --epochs "$PFGR_VALUE_EPOCHS" --batch-size "$PFGR_VALUE_BATCH_SIZE" --output-root "$OUTPUT_ROOT" \
    --run-name "R6-v222-$PFGR_RUN_ID" --device "$PFGR_DEVICE" --no-amp
fi
for value_variant in 126 270 366; do
  value_checkpoint="$OUTPUT_ROOT/R6-v${value_variant}-$PFGR_RUN_ID/value.pt"
  require_artifact "$value_checkpoint"
  "$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite value-evaluate \
    --config "$REPO_ROOT/configs/pfgr_lite/main.json" \
    --data-root "$BRATS21_ROOT" \
    --bank-index "$PFGR_BANK_INDEX" --checkpoint "$PFGR_BASE_CHECKPOINT" \
    --value-checkpoint "$value_checkpoint" --output-root "$OUTPUT_ROOT" \
    --split-file "$BASELINE_SPLIT" --roles-file "$PFGR_ROLES" \
    --run-name "R6-eval-v${value_variant}-$PFGR_RUN_ID" --device "$PFGR_DEVICE" --no-amp
done

# Mỗi lần value-evaluate phải tạo value_evaluate_pairs.json; join này xác nhận
# rằng V126/V270/V366 giữ cùng bank row/action/context, thay vì so sánh ba tập
# đã lọc khác nhau. Không chạy join nếu thiếu một biến thể hoặc một file.
PFGR_V126_PAIRS="$OUTPUT_ROOT/R6-eval-v126-$PFGR_RUN_ID/value_evaluate_pairs.json"
PFGR_V270_PAIRS="$OUTPUT_ROOT/R6-eval-v270-$PFGR_RUN_ID/value_evaluate_pairs.json"
PFGR_V366_PAIRS="$OUTPUT_ROOT/R6-eval-v366-$PFGR_RUN_ID/value_evaluate_pairs.json"
PFGR_R6_PAIR_JOIN_DIR="$OUTPUT_ROOT/R6-v-paired-$PFGR_RUN_ID"
PFGR_R6_PAIR_JOIN="$PFGR_R6_PAIR_JOIN_DIR/value_evaluate_pairs.json"
require_file "$PFGR_V126_PAIRS" V126-value_evaluate_pairs.json
require_file "$PFGR_V270_PAIRS" V270-value_evaluate_pairs.json
require_file "$PFGR_V366_PAIRS" V366-value_evaluate_pairs.json
mkdir "$PFGR_R6_PAIR_JOIN_DIR"
"$POINT_GUIDED_PYTHON" - "$PFGR_R6_PAIR_JOIN" "$PFGR_V126_PAIRS" "$PFGR_V270_PAIRS" "$PFGR_V366_PAIRS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
source_paths = tuple(Path(value) for value in sys.argv[2:])
if output_path.exists():
    raise SystemExit(f"refuse overwrite paired V join: {output_path}")
payloads = [json.loads(path.read_text(encoding="utf-8")) for path in source_paths]
if any(payload.get("schema_version") != "pfgr-lite-value-evaluation-pairs-v1" for payload in payloads):
    raise SystemExit("value_evaluate_pairs schema mismatch")
if any(payload.get("same_bank") is not True for payload in payloads):
    raise SystemExit("V pair source is not marked same_bank")
if [payload.get("input_variant") for payload in payloads] != [126, 270, 366]:
    raise SystemExit("expected V126/V270/V366 in fixed order")
bank_hashes = {payload.get("bank_manifest_hash") for payload in payloads}
if len(bank_hashes) != 1 or None in bank_hashes:
    raise SystemExit("V pair source bank manifests differ")
maps = []
for payload in payloads:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("V pair source has no rows")
    maps.append({row["row_key"]: row for row in rows})
keys = set(maps[0])
if any(set(rows) != keys for rows in maps[1:]):
    raise SystemExit("V pair row-key sets differ; same-bank join is invalid")
identity_fields = ("row_hash", "row_id", "shard", "offset", "subject_id", "context_id", "state_version", "point_id", "action_id", "proposal_hash", "state_digest", "group_key", "measured_rank")
for key in sorted(keys):
    reference = maps[0][key]
    for variant in maps[1:]:
        if any(variant[key].get(field) != reference.get(field) for field in identity_fields):
            raise SystemExit(f"immutable row identity differs for row_key={key}")
row_digest = hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()
joined = {
    "schema_version": "pfgr-lite-value-evaluation-join-v1",
    "same_bank": True,
    "bank_manifest_hash": next(iter(bank_hashes)),
    "input_variants": [126, 270, 366],
    "row_count": len(keys),
    "group_counts": [int(payload.get("group_count", 0)) for payload in payloads],
    "row_key_set_sha256": row_digest,
    "source_paths": [str(path) for path in source_paths],
    "teacher_calls": 0,
    "target_volume_reads": 0,
    "join_status": "PASS",
}
with output_path.open("x", encoding="utf-8") as handle:
    json.dump(joined, handle, sort_keys=True, indent=2)
    handle.write("\n")
print(json.dumps({"join_status": "PASS", "row_count": len(keys), "bank_manifest_hash": joined["bank_manifest_hash"]}, sort_keys=True))
PY
```

**Metrics.** Same-bank manifest/row hash, scale, fit rows/subjects, MSE vs
constant control, sign/rank/top-1 subset regret và V fit/evaluate call
counters (teacher/U/D/target phải zero). **SOFTWARE_PASS** là cached-only
fit/eval và joins đúng; **FAIL** là bank/V/producer/role mismatch hoặc hidden
teacher; scientific **INCONCLUSIVE** cho synthetic/underpowered pilot. Chọn
variant trước calibration; V-only refit invalidates calibration, không tự tune
trên final cohort.

## R7 -- train-only calibration, confirmation và adaptive gate

**Mục đích.** Forced K4 traces hoàn tất trước target; fit/allowance roles tách
producer-fit, calibration-fit, calibration-allowance; confirmation exact hoặc
fixed-Q độc lập. **Tiền đề.** R4 producer, R5 bank, R6 V366 và review receipt
human-authored đã khớp `config_hash/cohort_hash`. Production cần tối thiểu 32
subject groups và 64 winner rows cho mỗi role; fixture nhỏ không được waive.

```bash
require_artifact "$PFGR_BASE_CHECKPOINT"
require_artifact "$PFGR_VALUE_CHECKPOINT"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite calibrate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --value-checkpoint "$PFGR_VALUE_CHECKPOINT" --teacher-mode exact_footprint \
  --max-subjects 64 --dry-manifest --run-name "R7-review-request-$PFGR_RUN_ID"
require_artifact "$OUTPUT_ROOT/R7-review-request-$PFGR_RUN_ID/review_context.json"
# Người review đọc review_context.json, điền schema receipt thật vào path này.
require_artifact "$PFGR_CALIBRATION_REVIEW"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite calibrate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --value-checkpoint "$PFGR_VALUE_CHECKPOINT" --teacher-mode exact_footprint \
  --review-receipt "$PFGR_CALIBRATION_REVIEW" --max-subjects 64 \
  --run-name "R7-$PFGR_RUN_ID"
```

`--evidence` chỉ được dùng trong synthetic diagnostic khi sealed evidence đã
được reviewer kiểm tra và ghi rõ engineering-only; không dùng để biến fixture
thành adaptive production.
**Files/metrics.** forced traces trước labels, calibration evidence, raw-unit
`a,b`, q90 allowance, margins/tolerance, winner rows, role/dependency hashes,
`calibration.json`, và `adaptive.pt` chỉ khi đủ evidence. **SOFTWARE_PASS** chỉ
khi adaptive artifact hợp lệ; **FAIL** stale/duplicate/role overlap/screening
label reused; **INCONCLUSIVE** khi thiếu minimum hoặc fixture synthetic.
Thiếu calibration không chặn fixed/random/parallel R8; không tạo adaptive giả.

## R8 -- fixed/random/parallel trước adaptive; mỗi K riêng

**Mục đích.** Chạy diagnostics bằng effective-policy loader, không yêu cầu
adaptive trước fixed controls; `static`/`noop` chỉ K0 vì budget khác là
zero-action lặp lại. Random, fixed_learned và parallel_topk chạy K1/K2/K4
riêng, bounded rõ; không tự sinh matrix lớn. **Tiền đề.** R4 producer và R6
V cho fixed/parallel; R7 adaptive chỉ khi `adaptive.pt` tồn tại.

```bash
require_artifact "$PFGR_BASE_CHECKPOINT"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --scenario static --budget 0 --max-subjects 2 --split-role validation \
  --run-name "R8-static-k0-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --scenario noop --budget 0 --max-subjects 2 --split-role validation \
  --run-name "R8-noop-k0-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --scenario random --budget 1 --max-subjects 2 --split-role validation \
  --run-name "R8-random-k1-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --scenario random --budget 2 --max-subjects 2 --split-role validation \
  --run-name "R8-random-k2-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
  --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
  --scenario random --budget 4 --max-subjects 2 --split-role validation \
  --run-name "R8-random-k4-$PFGR_RUN_ID"
if [ -f "$PFGR_VALUE_CHECKPOINT" ]; then
  for policy_case in fixed_learned parallel_topk; do
    for budget in 1 2 4; do
      "$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
        --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_BASE_CHECKPOINT" \
        --value-checkpoint "$PFGR_VALUE_CHECKPOINT" --scenario "$policy_case" \
        --budget "$budget" --max-subjects 2 --split-role validation \
        --run-name "R8-${policy_case}-k${budget}-$PFGR_RUN_ID"
    done
  done
else
  echo "thiếu V; fixed_learned/parallel là INCONCLUSIVE, vẫn giữ random controls"
fi
if [ -f "$PFGR_CALIBRATED_CHECKPOINT" ]; then
  for budget in 1 2 4; do
    "$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
      --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_CALIBRATED_CHECKPOINT" \
      --value-checkpoint "$PFGR_VALUE_CHECKPOINT" --scenario adaptive \
      --budget "$budget" --max-subjects 2 --split-role validation \
      --run-name "R8-adaptive-k${budget}-$PFGR_RUN_ID"
  done
else
  echo "chưa có adaptive.pt; adaptive là INCONCLUSIVE và bị bỏ qua"
fi
```

**Metrics.** Mỗi receipt ghi scenario/K/effective-policy hash, z0/final
absolute và signed deltas, repeats, K bins, STOP reason, useful/harmful/neutral
denominators, measured candidate coverage, action identity, telescoping (fixed-Q
giữ MC uncertainty), parallel joint gain và interaction
`joint-sum(individual_initial)`. **SOFTWARE_PASS** khi run gọi đúng loader và
identity; **FAIL** target leak/stale identity/missing command; scientific
**INCONCLUSIVE** nếu thiếu adaptive/V/cohort/power. **Tiếp:** R9 chỉ sau
review; không xem K0 là K1, không chạy budget ngoài `{0,1,2,4}`.

## R9 -- matched final evaluation, điểm dừng bắt buộc

**Mục đích.** Chỉ chạy cohort/seed đã đóng băng sau review receipt; lần đầu
đọc test target là human gate. **Tiền đề.** R8 controls, calibrated bundle,
V366, roles/split/config hash và `PFGR_FINAL_REVIEW` do người ký.

```bash
if [ ! -f "$PFGR_CALIBRATED_CHECKPOINT" ]; then
  echo "không có adaptive.pt; R9 STOP, scientific INCONCLUSIVE"
else
  "$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
    --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_CALIBRATED_CHECKPOINT" \
    --value-checkpoint "$PFGR_VALUE_CHECKPOINT" --scenario adaptive --budget 4 \
    --split-role test --max-subjects 2 --dry-manifest \
    --run-name "R9-review-request-$PFGR_RUN_ID"
  require_artifact "$OUTPUT_ROOT/R9-review-request-$PFGR_RUN_ID/review_context.json"
  # Người review ký PFGR_FINAL_REVIEW sau khi đối chiếu cohort/hash thật.
  require_artifact "$PFGR_FINAL_REVIEW"
  "$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite evaluate "${PFGR_COMMON[@]}" \
    --roles-file "$PFGR_ROLES" --checkpoint "$PFGR_CALIBRATED_CHECKPOINT" \
    --value-checkpoint "$PFGR_VALUE_CHECKPOINT" --scenario adaptive --budget 4 \
    --split-role test --max-subjects 2 --review-receipt "$PFGR_FINAL_REVIEW" \
    --run-name "R9-reviewed-$PFGR_RUN_ID"
fi
```

**Files/metrics.** `paired_subjects.jsonl`, `action_metrics.jsonl`,
`metrics.json`, `effective_policy.json`, `receipt.json`, review/cohort hashes,
absolute/incremental MAE/PSNR/SSIM/Charbonnier, mask/range/denominators, paired
CI và same-work controls. **PASS** chỉ khi review/hash/cohort đúng và declared
margin CI; **FAIL** review mismatch hoặc target access trước gate; **INCONCLUSIVE**
underpowered/missing metric/unknown candidate scope. **STOP** sau run này để
Astra/user review; không tự đổi seed/cohort hay chạy test tiếp.

## R10 -- interrupted/resumed stage, cached-V continuation và package

**Mục đích.** Chứng minh resume thật từ interrupted update 1 sang continuation
2 với 2 subjects, không chỉ load/print; kiểm cached V incomplete resume khi
artifact thực tồn tại; package allow-list. S2 bank interrupted không append,
phải restart run-name mới như R5. **Tiền đề.** R1/R5/R8/R9 receipt tương ứng;
resume giữ config/split/producer/optimizer/RNG/input manifest.

```bash
PFGR_R10_INTERRUPTED_DIR="$OUTPUT_ROOT/R10-interrupted-$PFGR_RUN_ID"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite smoke --synthetic \
  --config "$REPO_ROOT/configs/pfgr_lite/synthetic.json" \
  --output-root "$OUTPUT_ROOT" --run-name "R10-interrupted-$PFGR_RUN_ID" \
  --device cpu --no-amp --max-subjects 2 --max-steps 1
PFGR_R10_INTERRUPTED_RESUME="$PFGR_R10_INTERRUPTED_DIR/resume.pt"
require_artifact "$PFGR_R10_INTERRUPTED_RESUME"
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite resume --synthetic \
  --config "$REPO_ROOT/configs/pfgr_lite/synthetic.json" \
  --resume-checkpoint "$PFGR_R10_INTERRUPTED_RESUME" \
  --output-root "$OUTPUT_ROOT" --run-name "R10-resumed-$PFGR_RUN_ID" \
  --device cpu --no-amp --max-subjects 2 --max-steps 2
```

Bounded synthetic retest cùng đúng argv đã PASS: smoke exit 0 tạo
`inference.pt`/`resume.pt`; resume exit 0 ghi `restored_updates=1`, `update=2`,
`optimizer_restored=true`, RNG `python/numpy/torch_cpu`,
`history_parent=prior_runtime`, `implicit_next_stage=false` và
`target_reads=0` ở resume. Receipt synthetic hiện ghi `subjects=1` dù giới hạn
`--max-subjects 2` vì fixture chỉ có một sample; đây là bằng chứng software
cho mechanics 1→2, chưa phải acceptance cohort hai subject. Chỉ đóng R10
production khi chạy lại cùng chuỗi trên split thật với receipt ghi
`counts.subjects=2`, so sánh weights/optimizer/RNG/cursor với uninterrupted
fixture và giữ traceback đầu tiên nếu có lỗi; không gọi synthetic này là khoa
học.

Cached V continuation là nhánh có điều kiện, không được bịa:

```bash
if [ -f "$PFGR_VALUE_RESUME" ]; then
  require_artifact "$PFGR_BANK_INDEX"
  "$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite resume \
    --config "$REPO_ROOT/configs/pfgr_lite/main.json" \
    --data-root "$BRATS21_ROOT" --split-file "$BASELINE_SPLIT" \
    --roles-file "$PFGR_ROLES" \
    --resume-checkpoint "$PFGR_VALUE_RESUME" --bank-index "$PFGR_BANK_INDEX" \
    --value-input 366 --epochs "$PFGR_VALUE_EPOCHS" \
    --batch-size "$PFGR_VALUE_BATCH_SIZE" --output-root "$OUTPUT_ROOT" \
    --run-name "R10-value-resumed-$PFGR_RUN_ID" --device "$PFGR_DEVICE" --no-amp \
    --max-steps 2
else
  echo "không có value-resume.pt chưa hoàn tất; cached-V continuation là PENDING"
fi
```

`--data-root`, `--split-file`, `--roles-file`, `--value-input`, `--epochs` và
`--batch-size` phải khớp chính xác lệnh `value-fit` đã tạo
`PFGR_VALUE_RESUME`; thiếu một identity production là lỗi, không phải lựa chọn
để resume từ filename. Chỉ khi `value_fit_incomplete.json`/`value-resume.pt`
thật tồn tại mới chạy nhánh này; output mới phải có `value.pt` hoặc một resume
incomplete mới với parent rõ ràng.

Trước khi package, lưu test output của lệnh bounded thực vào một run directory
riêng (không dùng dry-manifest). Output này không phải science result nhưng giúp
reviewer truy ngược lệnh/test đã chạy:

```bash
PFGR_TEST_OUTPUT_DIR="$OUTPUT_ROOT/R10-test-output-$PFGR_RUN_ID"
mkdir "$PFGR_TEST_OUTPUT_DIR"
{
  echo "argv: $POINT_GUIDED_PYTHON -m pytest -q tests/features/point_guided/pfgr_lite --tb=short"
  "$POINT_GUIDED_PYTHON" -m pytest -q tests/features/point_guided/pfgr_lite --tb=short
} > "$PFGR_TEST_OUTPUT_DIR/test_output.txt" 2>&1
echo "test output: $PFGR_TEST_OUTPUT_DIR/test_output.txt"

# Chỉ truyền các run directory đã tồn tại; path vắng là optional/incomplete,
# phải ghi chú rõ thay vì tạo manifest giả. --run-dir là repeatable và package
# chỉ copy allow-list metadata/metrics/test output, không copy checkpoint/raw bank.
PFGR_PACKAGE_CANDIDATES=(
  "$PFGR_R0_DIR"
  "$OUTPUT_ROOT/R0-runbook-check-$PFGR_RUN_ID"
  "$PFGR_R1_SYNTH_DIR" "$PFGR_R1_REAL_DIR"
  "$OUTPUT_ROOT/R2-$PFGR_RUN_ID"
  "$PFGR_STATIC_DIR" "$PFGR_R4_DIR" "$OUTPUT_ROOT/R4-u-only-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R4-static-k0-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R4-noop-k0-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R4-random-k1-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R4-random-k2-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R4-random-k4-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R4-u-only-random-k2-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R4-oracle-sampled1-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R4-oracle-greedy-k2-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R4-oracle-exact-subset8-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R4-oracle-all-n-reviewed-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R4-paired-$PFGR_RUN_ID"
  "$PFGR_R5_DIR" "$OUTPUT_ROOT/R5-verify-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R6-v126-$PFGR_RUN_ID" "$OUTPUT_ROOT/R6-v270-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R6-v366-$PFGR_RUN_ID" "$OUTPUT_ROOT/R6-v222-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R6-eval-v126-$PFGR_RUN_ID" "$OUTPUT_ROOT/R6-eval-v270-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R6-eval-v366-$PFGR_RUN_ID" "$OUTPUT_ROOT/R6-eval-v222-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R6-v-paired-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R7-review-request-$PFGR_RUN_ID" "$OUTPUT_ROOT/R7-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R8-static-k0-$PFGR_RUN_ID" "$OUTPUT_ROOT/R8-noop-k0-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R8-random-k1-$PFGR_RUN_ID" "$OUTPUT_ROOT/R8-random-k2-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R8-random-k4-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R8-fixed_learned-k1-$PFGR_RUN_ID" "$OUTPUT_ROOT/R8-fixed_learned-k2-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R8-fixed_learned-k4-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R8-parallel_topk-k1-$PFGR_RUN_ID" "$OUTPUT_ROOT/R8-parallel_topk-k2-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R8-parallel_topk-k4-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R8-adaptive-k1-$PFGR_RUN_ID" "$OUTPUT_ROOT/R8-adaptive-k2-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R8-adaptive-k4-$PFGR_RUN_ID"
  "$OUTPUT_ROOT/R9-review-request-$PFGR_RUN_ID" "$PFGR_REVIEWED_RUN_DIR"
  "$PFGR_TEST_OUTPUT_DIR"
)
PFGR_PACKAGE_ARGS=()
for candidate in "${PFGR_PACKAGE_CANDIDATES[@]}"; do
  if [ -d "$candidate" ]; then
    PFGR_PACKAGE_ARGS+=(--run-dir "$candidate")
  else
    echo "package bỏ qua artifact optional chưa tồn tại: $candidate"
  fi
done
test "${#PFGR_PACKAGE_ARGS[@]}" -gt 0 || { echo "không có run directory để package" >&2; exit 2; }
"$POINT_GUIDED_PYTHON" -m smagm.cli.pfgr_lite package \
  "${PFGR_PACKAGE_ARGS[@]}" --output-root "$OUTPUT_ROOT" \
  --run-name "R10-evidence-$PFGR_RUN_ID"
```

**Files/metrics.** Resume receipt phải có parent, stage/runtime cursor,
optimizer groups/moments, Python/NumPy/Torch/device RNG, input manifest hash,
actual update 1→2 và `implicit_next_stage=false`; so sánh weights/schedule với
uninterrupted fixture. Package `manifest.json` phải liệt kê các run directory
R0 provenance/tests, R2 benchmark, R4 controls/oracle/paired headroom, R5
bank/verify, R6 V receipts/paired join, R7 calibration, R8 controls, R9
reviewed result và test output nào thực sự tồn tại;
run thiếu được ghi exclusion/incomplete, không được tạo giả. Manifest chỉ
whitelist config/policy/
source/weights provenance/split/roles/receipts/metrics/action rows/benchmark/
test output/traceback/W&B ID thật; loại patient volumes, raw target,
predictions, checkpoints, secrets và raw bank shards. **PASS** khi weights,
optimizer/RNG/cursor khớp và allow-list đúng; **FAIL** stale/schema/split/RNG,
overwrite hoặc package leak; **INCONCLUSIVE** nếu cached V artifact chưa có.
**Tiếp/STOP:** giữ traceback và parent khi lỗi, không đánh dấu interrupted là
success, không tự chuyển stage.

### Bảng resource pilot bắt buộc

Mọi pilot phải giữ một dòng resource thật cho mỗi run, không điền số theo
ước tính. R1 ghi `inference`/`resume`/stage runtime và MedicalNet traversal;
R2 ghi cùng-work reference/optimized với `state_count`, `action_count`,
`candidate_count`, `query_count`, `D`/decoder calls, valid/padded support,
clone bytes, cache-build/cold/warm, allocated/reserved memory và GPU list;
R5 ghi bank shard/index bytes, rows, Q, replay và hash; R6 ghi V rows,
descriptor width (126/270/366), fit/evaluate calls và target/U/D calls phải
zero; R8/R9 ghi inference/decode, staged memory, action/STOP/terminal counts.
Tách hai cột `same_work` (cùng state/action/voxel/Q/dtype/chunk và full-write
reference) và `less_work` (giảm Q/state/chunk) trong `benchmark.json`; chỉ cột
đầu dùng cho parity, cột sau không được gọi speedup nếu chưa có đo ổn định.

## Troubleshooting và format trả về team

- **OOM:** giảm `--decode-chunk-size`, `--candidate-chunk-size`, `--max-states`
  hoặc `--query-count` rồi ghi thay đổi vào receipt/`changes.json`; không đổi
  precision, mask, threshold hay scope ngầm. Benchmark same-work phải giữ
  chunks/Q/action list như reference; nếu không, ghi reduced-work.
- **K0/zero/no legal:** K0 phải có zero proposal/V work nhưng vẫn decode Z0;
  phân biệt `budget`, `low_gain`, `no_legal_action`; no-op là correction bằng 0,
  không phải cold U/V. Kmax K4 dừng budget sau write thứ tư, không đoán fifth.
- **Không có U gradient/frozen-D:** kiểm `requires_grad`, nonzero input
  gradient, frozen parameter/BN hash và stage arm; fail closed, không bật
  `inference_mode` hay claim spectral utility từ random projector.
- **Không có V/thiếu calibration:** vẫn chạy static/noop/random/parallel theo
  R8; fixed/adaptive ghi INCONCLUSIVE hoặc skip dependency, không tái tạo
  policy/cached V từ filename.
- **Gain noisy/sign/confirmation:** giữ raw signed g, benefit/harm/neutral,
  M, Q, SE/variance và seed; dùng confirmation stream độc lập; negative không
  được relabel positive hoặc clip. Screening subset không thành all-N.
- **Bank/split/checkpoint/role mismatch:** dừng, giữ artifact/first traceback,
  kiểm producer compatibility, split hash, role digest, scale/lattice/writer
  hash và MedicalNet SHA/adaptation thực; không resplit hoặc thay checkpoint.
- **Parity gần ngưỡng/no speedup:** chạy FP64 exact tiny case, ghi max query/
  prediction/gain/gradient error và near-tie; cold/warm sync/CPU timing noisy
  là INCONCLUSIVE, không hứa speedup. Same-work và less-work phải tách.
- **Interruption/W&B:** resume only with strict runtime envelope; S2 bank
  restart run mới; W&B offline/local vẫn hợp lệ nhưng URL chỉ ghi khi thật.
- **Policy parity/parallel:** verify same effective-policy hash, stored action
  IDs/deltas, original initial state, no U rerun/rebase; parallel interaction
  là joint minus individual initial gains, không phải telescoping sequential.

Mỗi handoff cho team phải trả về: exact `argv`, exit code, first traceback (nếu
có), source SHA + dirty diff hash, resolved config/policy/weights/adaptation
hash, baseline split + roles/cohort extraction command, bank/index/scale,
calibration history, paired artifacts, operation counters/resources,
`scientific_status`, và W&B URL thật hoặc `null` với reason. CPU synthetic pass
chỉ là software evidence; CUDA/AMP/real/pretrained/patient/reconstruction
claims luôn **PENDING** cho Astra/user review.
