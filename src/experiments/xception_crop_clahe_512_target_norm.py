# xception_512_target_norm.py
# Xception 512x512 + weak augmentation + target normalization
# 실행: python .\xception_512_target_norm.py

from pathlib import Path
import random
import gc
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm
import os

PROJECT_DIR = Path(__file__).resolve().parent
TORCH_HOME_DIR = PROJECT_DIR / ".torch_cache"
TORCH_HUB_DIR = TORCH_HOME_DIR / "hub"

TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TORCH_HOME"] = str(TORCH_HOME_DIR)
torch.hub.set_dir(str(TORCH_HUB_DIR))

print("TORCH_HOME:", os.environ["TORCH_HOME"])
print("PyTorch Hub 캐시:", torch.hub.get_dir())


# ============================================================
# 0. 기본 설정
# ============================================================

SEED = 42
EXPERIMENT_NAME = "xception_crop_clahe_512_target_norm"

IMAGE_SIZE = 512

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_WORKERS = 0

MAX_EPOCHS = 35
EARLY_STOPPING_PATIENCE = 8

BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4

# 최종 모델 확정 전에는 False 유지
RUN_TEST_AFTER_TRAINING = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("Experiment:", EXPERIMENT_NAME)
print("Device:", device)
print("PyTorch:", torch.__version__)
print("timm:", timm.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 80)


# ============================================================
# 1. 경로 설정
# ============================================================

DATA_ROOT = Path(__file__).resolve().parent

TRAIN_CSV_PATH = (
    DATA_ROOT
    / "cropped_dataset_clahe"
    / "train_filtered.csv"
)

TRAIN_IMAGE_DIR = (
    DATA_ROOT
    / "cropped_dataset_clahe"
    / "train"
)

VAL_CSV_PATH = (
    DATA_ROOT
    / "boneage-validation-dataset"
    / "Validation Dataset.csv"
)

VAL_IMAGE_DIR = (
    DATA_ROOT
    / "cropped_dataset_clahe"
    / "val"
)

TEST_CSV_PATH = (
    DATA_ROOT
    / "Bone Age Test Set"
    / "Bone age ground truth.csv"
)

TEST_IMAGE_DIR = (
    DATA_ROOT
    / "cropped_dataset_clahe"
    / "test"
)

MODEL_DIR = DATA_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / f"best_{EXPERIMENT_NAME}.pt"
HISTORY_PATH = MODEL_DIR / f"history_{EXPERIMENT_NAME}.csv"
VAL_PRED_PATH = MODEL_DIR / f"val_predictions_{EXPERIMENT_NAME}.csv"
TEST_PRED_PATH = MODEL_DIR / f"test_predictions_{EXPERIMENT_NAME}.csv"


def check_path(name: str, path: Path) -> None:
    print(f"{name}: {path.exists()} | {path}")
    if not path.exists():
        raise FileNotFoundError(f"{name} 경로 없음: {path}")


check_path("Train CSV", TRAIN_CSV_PATH)
check_path("Train image dir", TRAIN_IMAGE_DIR)
check_path("Validation CSV", VAL_CSV_PATH)
check_path("Validation image dir", VAL_IMAGE_DIR)

if RUN_TEST_AFTER_TRAINING:
    check_path("Test CSV", TEST_CSV_PATH)
    check_path("Test image dir", TEST_IMAGE_DIR)


# ============================================================
# 2. CSV 정리
# ============================================================

from pandas.api.types import is_bool_dtype, is_numeric_dtype


def normalize_boneage_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe.columns = (
        dataframe.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    rename_map = {
        "image_id": "id",
        "case_id": "id",
        "bone_age": "boneage",
        "bone_age_months": "boneage",
        "ground_truth_bone_age_months": "boneage",
        "sex": "male",
        "gender": "male",
    }

    dataframe = dataframe.rename(columns=rename_map)

    required_columns = {"id", "boneage", "male"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"필수 컬럼 누락: {missing_columns}")

    dataframe["id"] = (
        dataframe["id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    dataframe["boneage"] = pd.to_numeric(
        dataframe["boneage"], errors="raise"
    ).astype(float)

    original_male = dataframe["male"]

    if is_bool_dtype(original_male):
        dataframe["male"] = original_male.astype(float)
    elif is_numeric_dtype(original_male):
        dataframe["male"] = pd.to_numeric(original_male, errors="coerce").astype(float)
    else:
        normalized_male = original_male.astype(str).str.strip().str.lower()
        dataframe["male"] = normalized_male.map(
            {
                "true": 1.0,
                "false": 0.0,
                "1": 1.0,
                "0": 0.0,
                "1.0": 1.0,
                "0.0": 0.0,
                "m": 1.0,
                "f": 0.0,
                "male": 1.0,
                "female": 0.0,
            }
        )

    if dataframe["male"].isna().any():
        failed_values = (
            original_male[dataframe["male"].isna()].astype(str).unique().tolist()
        )
        raise ValueError(f"성별 변환 실패 값: {failed_values}")

    invalid_values = set(dataframe["male"].unique()) - {0.0, 1.0}
    if invalid_values:
        raise ValueError(f"성별에 0/1 이외 값 존재: {invalid_values}")

    return dataframe[["id", "boneage", "male"]].reset_index(drop=True)


train_df = normalize_boneage_dataframe(pd.read_csv(TRAIN_CSV_PATH))
val_df = normalize_boneage_dataframe(pd.read_csv(VAL_CSV_PATH))

test_df = normalize_boneage_dataframe(pd.read_csv(TEST_CSV_PATH)) if RUN_TEST_AFTER_TRAINING else None

print("\n[CSV]")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
if RUN_TEST_AFTER_TRAINING:
    print("Test:", test_df.shape)


# ============================================================
# 3. Target 정규화 통계
# ============================================================

TARGET_MEAN = float(train_df["boneage"].mean())
TARGET_STD = float(train_df["boneage"].std())

print("\n[Target Normalization]")
print("TARGET_MEAN:", TARGET_MEAN)
print("TARGET_STD:", TARGET_STD)


# ============================================================
# 4. 이미지 경로 연결
# ============================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def build_image_index(image_dir: Path) -> dict:
    image_index = {}
    for image_path in image_dir.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_id = image_path.stem
            if image_id in image_index:
                raise ValueError(f"중복 이미지 ID: {image_id}")
            image_index[image_id] = str(image_path)
    print(f"{image_dir.name}: {len(image_index)} images")
    return image_index


def attach_image_paths(dataframe: pd.DataFrame, image_index: dict, dataset_name: str) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["image_path"] = dataframe["id"].map(image_index)
    missing_df = dataframe[dataframe["image_path"].isna()]
    print(f"{dataset_name} missing images:", len(missing_df))
    if len(missing_df) > 0:
        print("missing examples:", missing_df["id"].head(10).tolist())
        raise FileNotFoundError(f"{dataset_name} 이미지 누락")
    return dataframe


train_image_index = build_image_index(TRAIN_IMAGE_DIR)
val_image_index = build_image_index(VAL_IMAGE_DIR)

train_df = attach_image_paths(train_df, train_image_index, "Train")
val_df = attach_image_paths(val_df, val_image_index, "Validation")

if RUN_TEST_AFTER_TRAINING:
    test_image_index = build_image_index(TEST_IMAGE_DIR)
    test_df = attach_image_paths(test_df, test_image_index, "Test")


def print_dataset_summary(name: str, dataframe: pd.DataFrame) -> None:
    print(f"\n[{name}]")
    print("count:", len(dataframe))
    print("boneage mean:", round(dataframe["boneage"].mean(), 3))
    print("boneage std:", round(dataframe["boneage"].std(), 3))
    print("boneage min:", dataframe["boneage"].min())
    print("boneage max:", dataframe["boneage"].max())
    print("male ratio:", round(dataframe["male"].mean(), 3))


print_dataset_summary("Train", train_df)
print_dataset_summary("Validation", val_df)
if RUN_TEST_AFTER_TRAINING:
    print_dataset_summary("Test", test_df)


# ============================================================
# 5. Transform
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            scale=(0.97, 1.03),
            interpolation=InterpolationMode.BICUBIC,
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


# ============================================================
# 6. Dataset / DataLoader
# ============================================================

class BoneAgeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform, target_mean: float, target_std: float):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
        image = self.transform(image)

        sex = torch.tensor([float(row["male"])], dtype=torch.float32)
        bone_age_month = float(row["boneage"])
        bone_age_norm = (bone_age_month - self.target_mean) / self.target_std

        return {
            "image": image,
            "sex": sex,
            "bone_age": torch.tensor(bone_age_norm, dtype=torch.float32),
            "bone_age_month": torch.tensor(bone_age_month, dtype=torch.float32),
            "id": str(row["id"]),
        }


train_dataset = BoneAgeDataset(train_df, train_transform, TARGET_MEAN, TARGET_STD)
val_dataset = BoneAgeDataset(val_df, eval_transform, TARGET_MEAN, TARGET_STD)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

if RUN_TEST_AFTER_TRAINING:
    test_dataset = BoneAgeDataset(test_df, eval_transform, TARGET_MEAN, TARGET_STD)
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
else:
    test_loader = None

print("\n[DataLoader]")
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

sample_batch = next(iter(train_loader))
print("Image shape:", sample_batch["image"].shape)
print("Sex shape:", sample_batch["sex"].shape)
print("Bone age norm shape:", sample_batch["bone_age"].shape)
print("Bone age month shape:", sample_batch["bone_age_month"].shape)
print("Bone age month examples:", sample_batch["bone_age_month"][:5])
print("Bone age norm examples:", sample_batch["bone_age"][:5])


# ============================================================
# 7. Model
# ============================================================

class XceptionLinearBoneAgeModel(nn.Module):
    def __init__(self, pretrained: bool = True, image_size: int = 512):
        super().__init__()

        self.backbone = timm.create_model(
            "legacy_xception",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        self.image_head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Flatten(),
        )

        self.sex_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
        )

        image_feature_dim = self._infer_image_feature_dim(image_size=image_size)
        print("image_feature_dim:", image_feature_dim)

        self.regressor = nn.Linear(image_feature_dim + 32, 1)

    def _infer_image_feature_dim(self, image_size: int) -> int:
        was_training_backbone = self.backbone.training
        was_training_head = self.image_head.training

        self.backbone.eval()
        self.image_head.eval()

        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, image_size, image_size)
            dummy_feature = self.backbone.forward_features(dummy_image)
            dummy_vector = self.image_head(dummy_feature)
            image_feature_dim = int(dummy_vector.shape[1])

        if was_training_backbone:
            self.backbone.train()
        if was_training_head:
            self.image_head.train()

        return image_feature_dim

    def forward(self, image, sex):
        feature_map = self.backbone.forward_features(image)
        image_feature = self.image_head(feature_map)
        sex_feature = self.sex_embedding(sex)
        combined_feature = torch.cat([image_feature, sex_feature], dim=1)
        prediction_norm = self.regressor(combined_feature).squeeze(1)
        return prediction_norm


gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model = XceptionLinearBoneAgeModel(pretrained=True, image_size=IMAGE_SIZE).to(device)

for parameter in model.parameters():
    parameter.requires_grad = True

total_parameters = sum(parameter.numel() for parameter in model.parameters())
trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

print("\n[Model]")
print("Model device:", next(model.parameters()).device)
print("Total parameters:", f"{total_parameters:,}")
print("Trainable parameters:", f"{trainable_parameters:,}")

model.eval()
with torch.no_grad():
    test_images = sample_batch["image"][:2].to(device)
    test_sexes = sample_batch["sex"][:2].to(device)
    feature_map = model.backbone.forward_features(test_images)
    conv_feature = model.image_head[0](feature_map)
    pooled_feature = model.image_head[2](model.image_head[1](conv_feature))
    image_vector = model.image_head[3](pooled_feature)
    sex_vector = model.sex_embedding(test_sexes)
    predictions_norm = model(test_images, test_sexes)
    predictions_month = predictions_norm * TARGET_STD + TARGET_MEAN

print("Xception feature:", feature_map.shape)
print("Conv feature:", conv_feature.shape)
print("Pool feature:", pooled_feature.shape)
print("Image vector:", image_vector.shape)
print("Sex vector:", sex_vector.shape)
print("Prediction norm:", predictions_norm.shape)
print("Prediction month:", predictions_month.shape)


# ============================================================
# 8. Train / Evaluate
# ============================================================

def train_one_epoch(model, data_loader, optimizer, criterion, device, target_mean: float, target_std: float):
    model.train()
    total_loss = 0.0
    total_absolute_error_month = 0.0
    total_samples = 0

    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        sexes = batch["sex"].to(device, non_blocking=True)
        targets_norm = batch["bone_age"].to(device, non_blocking=True)
        targets_month = batch["bone_age_month"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions_norm = model(images, sexes)
        loss = criterion(predictions_norm, targets_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        predictions_month = predictions_norm.detach() * target_std + target_mean
        batch_size = targets_norm.size(0)
        total_loss += loss.item() * batch_size
        total_absolute_error_month += torch.abs(predictions_month - targets_month).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "mae": total_absolute_error_month / total_samples,
    }


def evaluate(model, data_loader, criterion, device, target_mean: float, target_std: float):
    model.eval()
    total_loss = 0.0
    total_absolute_error_month = 0.0
    total_squared_error_month = 0.0
    total_samples = 0
    all_predictions_month = []
    all_targets_month = []
    all_predictions_norm = []
    all_targets_norm = []
    all_ids = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device, non_blocking=True)
            sexes = batch["sex"].to(device, non_blocking=True)
            targets_norm = batch["bone_age"].to(device, non_blocking=True)
            targets_month = batch["bone_age_month"].to(device, non_blocking=True)

            predictions_norm = model(images, sexes)
            loss = criterion(predictions_norm, targets_norm)
            predictions_month = predictions_norm * target_std + target_mean
            errors_month = predictions_month - targets_month

            batch_size = targets_norm.size(0)
            total_loss += loss.item() * batch_size
            total_absolute_error_month += torch.abs(errors_month).sum().item()
            total_squared_error_month += torch.square(errors_month).sum().item()
            total_samples += batch_size

            all_predictions_month.extend(predictions_month.cpu().numpy())
            all_targets_month.extend(targets_month.cpu().numpy())
            all_predictions_norm.extend(predictions_norm.cpu().numpy())
            all_targets_norm.extend(targets_norm.cpu().numpy())
            all_ids.extend(batch["id"])

    return {
        "loss": total_loss / total_samples,
        "mae": total_absolute_error_month / total_samples,
        "rmse": np.sqrt(total_squared_error_month / total_samples),
        "predictions": np.asarray(all_predictions_month),
        "targets": np.asarray(all_targets_month),
        "predictions_norm": np.asarray(all_predictions_norm),
        "targets_norm": np.asarray(all_targets_norm),
        "ids": all_ids,
    }


# ============================================================
# 9. Optimizer / Scheduler
# ============================================================

criterion = nn.L1Loss()

optimizer = torch.optim.AdamW(
    [
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.image_head.parameters(), "lr": HEAD_LR},
        {"params": model.sex_embedding.parameters(), "lr": HEAD_LR},
        {"params": model.regressor.parameters(), "lr": HEAD_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
)


# ============================================================
# 10. Training Loop
# ============================================================

best_val_mae = float("inf")
no_improvement_count = 0
history = []

print("\n[Training Start]")
print("BEST_MODEL_PATH:", BEST_MODEL_PATH)
print("HISTORY_PATH:", HISTORY_PATH)
print("=" * 80)

start_time = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    epoch_start_time = time.time()

    train_result = train_one_epoch(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        target_mean=TARGET_MEAN,
        target_std=TARGET_STD,
    )

    val_result = evaluate(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        device=device,
        target_mean=TARGET_MEAN,
        target_std=TARGET_STD,
    )

    scheduler.step(val_result["mae"])

    backbone_lr = optimizer.param_groups[0]["lr"]
    head_lr = optimizer.param_groups[1]["lr"]
    epoch_time = time.time() - epoch_start_time

    history.append(
        {
            "epoch": epoch,
            "train_loss_norm": train_result["loss"],
            "train_mae": train_result["mae"],
            "val_loss_norm": val_result["loss"],
            "val_mae": val_result["mae"],
            "val_rmse": val_result["rmse"],
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "epoch_time_sec": epoch_time,
        }
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Train MAE: {train_result['mae']:.3f}개월 | "
        f"Val MAE: {val_result['mae']:.3f}개월 | "
        f"Val RMSE: {val_result['rmse']:.3f}개월 | "
        f"Norm Loss: {val_result['loss']:.4f} | "
        f"Backbone LR: {backbone_lr:.2e} | "
        f"Head LR: {head_lr:.2e} | "
        f"Time: {epoch_time:.1f}s"
    )

    if val_result["mae"] < best_val_mae:
        best_val_mae = val_result["mae"]
        no_improvement_count = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mae": best_val_mae,
                "history": history,
                "experiment_name": EXPERIMENT_NAME,
                "image_size": IMAGE_SIZE,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "backbone_lr": BACKBONE_LR,
                "head_lr": HEAD_LR,
                "weight_decay": WEIGHT_DECAY,
                "target_mean": TARGET_MEAN,
                "target_std": TARGET_STD,
                "target_normalized": True,
                "model_name": "XceptionLinearBoneAgeModel",
            },
            BEST_MODEL_PATH,
        )
        print(f"최적 모델 저장: {best_val_mae:.3f}개월")
    else:
        no_improvement_count += 1
        print("성능 미개선:", f"{no_improvement_count}/{EARLY_STOPPING_PATIENCE}")

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")

    if no_improvement_count >= EARLY_STOPPING_PATIENCE:
        print("Early stopping")
        break


total_time = time.time() - start_time
print("=" * 80)
print("Training finished")
print("Best Validation MAE:", f"{best_val_mae:.3f}개월")
print("Total time:", f"{total_time / 60:.1f} min")
print("=" * 80)


# ============================================================
# 11. Best Model Load + Validation Prediction Save
# ============================================================

checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

target_mean = float(checkpoint["target_mean"])
target_std = float(checkpoint["target_std"])

final_val_result = evaluate(
    model=model,
    data_loader=val_loader,
    criterion=criterion,
    device=device,
    target_mean=target_mean,
    target_std=target_std,
)

print("\n[Best Validation Result]")
print("Best epoch:", checkpoint["epoch"])
print("Saved best Val MAE:", checkpoint["best_val_mae"])
print("Reloaded Val MAE:", final_val_result["mae"])
print("Reloaded Val RMSE:", final_val_result["rmse"])

val_result_df = pd.DataFrame(
    {
        "id": final_val_result["ids"],
        "actual_bone_age": final_val_result["targets"],
        "predicted_bone_age": final_val_result["predictions"],
        "actual_bone_age_norm": final_val_result["targets_norm"],
        "predicted_bone_age_norm": final_val_result["predictions_norm"],
    }
)

val_result_df["error"] = val_result_df["predicted_bone_age"] - val_result_df["actual_bone_age"]
val_result_df["absolute_error"] = val_result_df["error"].abs()

val_result_df.to_csv(VAL_PRED_PATH, index=False, encoding="utf-8-sig")
print("Validation prediction saved:", VAL_PRED_PATH)


# ============================================================
# 12. Optional Test
# ============================================================

if RUN_TEST_AFTER_TRAINING:
    test_result = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    print("\n[Test Result]")
    print(f"Test MAE: {test_result['mae']:.3f}개월")
    print(f"Test RMSE: {test_result['rmse']:.3f}개월")

    test_result_df = pd.DataFrame(
        {
            "id": test_result["ids"],
            "actual_bone_age": test_result["targets"],
            "predicted_bone_age": test_result["predictions"],
            "actual_bone_age_norm": test_result["targets_norm"],
            "predicted_bone_age_norm": test_result["predictions_norm"],
        }
    )

    test_result_df["error"] = test_result_df["predicted_bone_age"] - test_result_df["actual_bone_age"]
    test_result_df["absolute_error"] = test_result_df["error"].abs()

    test_result_df.to_csv(TEST_PRED_PATH, index=False, encoding="utf-8-sig")
    print("Test prediction saved:", TEST_PRED_PATH)