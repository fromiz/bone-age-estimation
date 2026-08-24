from pathlib import Path
import random
import gc
import time
import os

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm

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

EXPERIMENT_NAME = "xception_weak_aug_512"

IMAGE_SIZE = 512

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

NUM_WORKERS = 0

MAX_EPOCHS = 35
EARLY_STOPPING_PATIENCE = 8

BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4

RUN_TEST_AFTER_TRAINING = False  # 최종 모델 확정 전에는 False 유지


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("Experiment:", EXPERIMENT_NAME)
print("Device:", device)
print("PyTorch:", torch.__version__)
print("timm:", timm.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 60)


# ============================================================
# 1. 경로 설정
# ============================================================

# 현재 xception_weak_aug_512.py가 있는 폴더 기준
DATA_ROOT = Path(__file__).resolve().parent

TRAIN_CSV_PATH = (
    DATA_ROOT
    / "boneage-training-dataset"
    / "train.csv"
)

TRAIN_IMAGE_DIR = (
    DATA_ROOT
    / "boneage-training-dataset"
    / "boneage-training-dataset"
)

VAL_CSV_PATH = (
    DATA_ROOT
    / "boneage-validation-dataset"
    / "Validation Dataset.csv"
)

VAL_IMAGE_DIR = (
    DATA_ROOT
    / "boneage-validation-dataset"
    / "boneage-validation-dataset"
)

TEST_CSV_PATH = (
    DATA_ROOT
    / "Bone Age Test Set"
    / "Bone age ground truth.csv"
)

TEST_IMAGE_DIR = (
    DATA_ROOT
    / "Bone Age Test Set"
    / "Test Set Images"
)

MODEL_DIR = DATA_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = (
    MODEL_DIR
    / f"best_{EXPERIMENT_NAME}.pt"
)

HISTORY_PATH = (
    MODEL_DIR
    / f"history_{EXPERIMENT_NAME}.csv"
)

VAL_PRED_PATH = (
    MODEL_DIR
    / f"val_predictions_{EXPERIMENT_NAME}.csv"
)

TEST_PRED_PATH = (
    MODEL_DIR
    / f"test_predictions_{EXPERIMENT_NAME}.csv"
)


def check_path(name, path):
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


def normalize_boneage_dataframe(dataframe):
    dataframe = dataframe.copy()

    dataframe.columns = (
        dataframe.columns
        .astype(str)
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
        dataframe["boneage"],
        errors="raise",
    ).astype(float)

    original_male = dataframe["male"]

    if is_bool_dtype(original_male):
        dataframe["male"] = original_male.astype(float)

    elif is_numeric_dtype(original_male):
        dataframe["male"] = pd.to_numeric(
            original_male,
            errors="coerce",
        ).astype(float)

    else:
        normalized_male = (
            original_male
            .astype(str)
            .str.strip()
            .str.lower()
        )

        dataframe["male"] = normalized_male.map({
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
        })

    if dataframe["male"].isna().any():
        failed_values = (
            original_male[dataframe["male"].isna()]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(f"성별 변환 실패 값: {failed_values}")

    invalid_values = set(dataframe["male"].unique()) - {0.0, 1.0}

    if invalid_values:
        raise ValueError(f"성별에 0/1 이외 값 존재: {invalid_values}")

    return dataframe[["id", "boneage", "male"]].reset_index(drop=True)


train_df = normalize_boneage_dataframe(
    pd.read_csv(TRAIN_CSV_PATH)
)

val_df = normalize_boneage_dataframe(
    pd.read_csv(VAL_CSV_PATH)
)

print("\n[CSV]")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)

if RUN_TEST_AFTER_TRAINING:
    test_df = normalize_boneage_dataframe(
        pd.read_csv(TEST_CSV_PATH)
    )
    print("Test:", test_df.shape)
else:
    test_df = None


# ============================================================
# 3. 이미지 경로 연결
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


def build_image_index(image_dir):
    image_index = {}

    for image_path in image_dir.rglob("*"):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            image_id = image_path.stem

            if image_id in image_index:
                raise ValueError(f"중복 이미지 ID: {image_id}")

            image_index[image_id] = str(image_path)

    print(f"{image_dir.name}: {len(image_index)} images")

    return image_index


def attach_image_paths(dataframe, image_index, dataset_name):
    dataframe = dataframe.copy()

    dataframe["image_path"] = dataframe["id"].map(image_index)

    missing_df = dataframe[dataframe["image_path"].isna()]

    print(f"{dataset_name} missing images:", len(missing_df))

    if len(missing_df) > 0:
        print(missing_df["id"].head(10).tolist())
        raise FileNotFoundError(f"{dataset_name} 이미지 누락")

    return dataframe


train_image_index = build_image_index(TRAIN_IMAGE_DIR)
val_image_index = build_image_index(VAL_IMAGE_DIR)

train_df = attach_image_paths(
    train_df,
    train_image_index,
    "Train",
)

val_df = attach_image_paths(
    val_df,
    val_image_index,
    "Validation",
)

if RUN_TEST_AFTER_TRAINING:
    test_image_index = build_image_index(TEST_IMAGE_DIR)

    test_df = attach_image_paths(
        test_df,
        test_image_index,
        "Test",
    )


def print_dataset_summary(name, dataframe):
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
# 4. Transform
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=InterpolationMode.BICUBIC,
    ),

    transforms.RandomAffine(
        degrees=5,
        translate=(0.03, 0.03),
        scale=(0.97, 1.03),
        interpolation=InterpolationMode.BICUBIC,
        fill=0,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=InterpolationMode.BICUBIC,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# ============================================================
# 5. Dataset / DataLoader
# ============================================================

class BoneAgeDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")

        image = self.transform(image)

        sex = torch.tensor(
            [float(row["male"])],
            dtype=torch.float32,
        )

        bone_age = torch.tensor(
            float(row["boneage"]),
            dtype=torch.float32,
        )

        return {
            "image": image,
            "sex": sex,
            "bone_age": bone_age,
            "id": str(row["id"]),
        }


train_dataset = BoneAgeDataset(
    train_df,
    train_transform,
)

val_dataset = BoneAgeDataset(
    val_df,
    eval_transform,
)

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
    test_dataset = BoneAgeDataset(
        test_df,
        eval_transform,
    )

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
print("Bone age shape:", sample_batch["bone_age"].shape)
print("Bone age examples:", sample_batch["bone_age"][:5])


# ============================================================
# 6. Model
# ============================================================

class XceptionLinearBoneAgeModel(nn.Module):
    def __init__(
        self,
        pretrained=True,
        image_size=512,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            "legacy_xception",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        self.image_head = nn.Sequential(
            nn.Conv2d(
                in_channels=2048,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=3,
            ),
            nn.Flatten(),
        )

        self.sex_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
        )

        image_feature_dim = self._infer_image_feature_dim(
            image_size=image_size
        )

        print("image_feature_dim:", image_feature_dim)

        self.regressor = nn.Linear(
            image_feature_dim + 32,
            1,
        )

    def _infer_image_feature_dim(self, image_size):
        was_training_backbone = self.backbone.training
        was_training_head = self.image_head.training

        self.backbone.eval()
        self.image_head.eval()

        with torch.no_grad():
            dummy_image = torch.zeros(
                1,
                3,
                image_size,
                image_size,
            )

            dummy_feature = self.backbone.forward_features(
                dummy_image
            )

            dummy_vector = self.image_head(
                dummy_feature
            )

            image_feature_dim = dummy_vector.shape[1]

        if was_training_backbone:
            self.backbone.train()

        if was_training_head:
            self.image_head.train()

        return image_feature_dim

    def forward(self, image, sex):
        feature_map = self.backbone.forward_features(image)

        image_feature = self.image_head(feature_map)

        sex_feature = self.sex_embedding(sex)

        combined_feature = torch.cat(
            [image_feature, sex_feature],
            dim=1,
        )

        prediction = self.regressor(combined_feature).squeeze(1)

        return prediction


gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

model = XceptionLinearBoneAgeModel(
    pretrained=True,
    image_size=IMAGE_SIZE,
).to(device)

for parameter in model.parameters():
    parameter.requires_grad = True


total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\n[Model]")
print("Model device:", next(model.parameters()).device)
print("Total parameters:", f"{total_parameters:,}")
print("Trainable parameters:", f"{trainable_parameters:,}")


# shape 확인
model.eval()

with torch.no_grad():
    test_images = sample_batch["image"][:2].to(device)
    test_sexes = sample_batch["sex"][:2].to(device)

    feature_map = model.backbone.forward_features(test_images)
    conv_feature = model.image_head[0](feature_map)
    pooled_feature = model.image_head[2](
        model.image_head[1](conv_feature)
    )
    image_vector = model.image_head[3](pooled_feature)
    sex_vector = model.sex_embedding(test_sexes)
    predictions = model(test_images, test_sexes)

print("Xception feature:", feature_map.shape)
print("Conv feature:", conv_feature.shape)
print("Pool feature:", pooled_feature.shape)
print("Image vector:", image_vector.shape)
print("Sex vector:", sex_vector.shape)
print("Prediction:", predictions.shape)


# ============================================================
# 7. Train / Evaluate
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    for batch_index, batch in enumerate(data_loader, start=1):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        sexes = batch["sex"].to(
            device,
            non_blocking=True,
        )

        targets = batch["bone_age"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        predictions = model(images, sexes)

        loss = criterion(predictions, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size

        total_absolute_error += (
            torch.abs(predictions.detach() - targets)
            .sum()
            .item()
        )

        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "mae": total_absolute_error / total_samples,
    }


def evaluate(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_squared_error = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            sexes = batch["sex"].to(
                device,
                non_blocking=True,
            )

            targets = batch["bone_age"].to(
                device,
                non_blocking=True,
            )

            predictions = model(images, sexes)

            loss = criterion(predictions, targets)

            errors = predictions - targets
            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size

            total_absolute_error += (
                torch.abs(errors)
                .sum()
                .item()
            )

            total_squared_error += (
                torch.square(errors)
                .sum()
                .item()
            )

            total_samples += batch_size

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                targets.cpu().numpy()
            )

            all_ids.extend(batch["id"])

    return {
        "loss": total_loss / total_samples,
        "mae": total_absolute_error / total_samples,
        "rmse": np.sqrt(total_squared_error / total_samples),
        "predictions": np.asarray(all_predictions),
        "targets": np.asarray(all_targets),
        "ids": all_ids,
    }


# ============================================================
# 8. Optimizer / Scheduler
# ============================================================

criterion = nn.L1Loss()

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.backbone.parameters(),
            "lr": BACKBONE_LR,
        },
        {
            "params": model.image_head.parameters(),
            "lr": HEAD_LR,
        },
        {
            "params": model.sex_embedding.parameters(),
            "lr": HEAD_LR,
        },
        {
            "params": model.regressor.parameters(),
            "lr": HEAD_LR,
        },
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
# 9. Training Loop
# ============================================================

best_val_mae = float("inf")
no_improvement_count = 0
history = []

print("\n[Training Start]")
print("BEST_MODEL_PATH:", BEST_MODEL_PATH)
print("HISTORY_PATH:", HISTORY_PATH)
print("=" * 60)

start_time = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    epoch_start_time = time.time()

    train_result = train_one_epoch(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    val_result = evaluate(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        device=device,
    )

    scheduler.step(val_result["mae"])

    backbone_lr = optimizer.param_groups[0]["lr"]
    head_lr = optimizer.param_groups[1]["lr"]

    epoch_time = time.time() - epoch_start_time

    history.append({
        "epoch": epoch,
        "train_loss": train_result["loss"],
        "train_mae": train_result["mae"],
        "val_loss": val_result["loss"],
        "val_mae": val_result["mae"],
        "val_rmse": val_result["rmse"],
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "epoch_time_sec": epoch_time,
    })

    print(
        f"Epoch {epoch:02d} | "
        f"Train MAE: {train_result['mae']:.3f}개월 | "
        f"Val MAE: {val_result['mae']:.3f}개월 | "
        f"Val RMSE: {val_result['rmse']:.3f}개월 | "
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
                "model_name": "XceptionLinearBoneAgeModel",
            },
            BEST_MODEL_PATH,
        )

        print(f"최적 모델 저장: {best_val_mae:.3f}개월")

    else:
        no_improvement_count += 1

        print(
            "성능 미개선:",
            f"{no_improvement_count}/{EARLY_STOPPING_PATIENCE}",
        )

    pd.DataFrame(history).to_csv(
        HISTORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if no_improvement_count >= EARLY_STOPPING_PATIENCE:
        print("Early stopping")
        break


total_time = time.time() - start_time

print("=" * 60)
print("Training finished")
print("Best Validation MAE:", f"{best_val_mae:.3f}개월")
print("Total time:", f"{total_time / 60:.1f} min")
print("=" * 60)


# ============================================================
# 10. Best Model Load + Validation Prediction Save
# ============================================================

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

final_val_result = evaluate(
    model=model,
    data_loader=val_loader,
    criterion=criterion,
    device=device,
)

print("\n[Best Validation Result]")
print("Best epoch:", checkpoint["epoch"])
print("Saved best Val MAE:", checkpoint["best_val_mae"])
print("Reloaded Val MAE:", final_val_result["mae"])
print("Reloaded Val RMSE:", final_val_result["rmse"])

val_result_df = pd.DataFrame({
    "id": final_val_result["ids"],
    "actual_bone_age": final_val_result["targets"],
    "predicted_bone_age": final_val_result["predictions"],
})

val_result_df["error"] = (
    val_result_df["predicted_bone_age"]
    - val_result_df["actual_bone_age"]
)

val_result_df["absolute_error"] = (
    val_result_df["error"].abs()
)

val_result_df.to_csv(
    VAL_PRED_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("Validation prediction saved:", VAL_PRED_PATH)


# ============================================================
# 11. Optional Test
# ============================================================

if RUN_TEST_AFTER_TRAINING:
    test_result = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\n[Test Result]")
    print(f"Test MAE: {test_result['mae']:.3f}개월")
    print(f"Test RMSE: {test_result['rmse']:.3f}개월")

    test_result_df = pd.DataFrame({
        "id": test_result["ids"],
        "actual_bone_age": test_result["targets"],
        "predicted_bone_age": test_result["predictions"],
    })

    test_result_df["error"] = (
        test_result_df["predicted_bone_age"]
        - test_result_df["actual_bone_age"]
    )

    test_result_df["absolute_error"] = (
        test_result_df["error"].abs()
    )

    test_result_df.to_csv(
        TEST_PRED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("Test prediction saved:", TEST_PRED_PATH)