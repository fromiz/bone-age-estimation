from pathlib import Path
import gc
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm

# 현재 프로젝트 안에 PyTorch 사전학습 가중치 캐시 생성
PROJECT_DIR = Path(__file__).resolve().parent
TORCH_HUB_DIR = PROJECT_DIR / ".torch_cache" / "hub"

TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(TORCH_HUB_DIR))

print("PyTorch Hub 캐시:", torch.hub.get_dir())


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("PyTorch version:", torch.__version__)
print("timm version:", timm.__version__)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

DATA_ROOT = Path(
    "./"
)

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
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "best_xception_weak_augmentation.pt"
)

HISTORY_PATH = (
    MODEL_DIR
    / "xception_weak_augmentation_history.csv"
)

TEST_RESULT_PATH = (
    MODEL_DIR
    / "xception_weak_augmentation_predictions.csv"
)

paths_to_check = {
    "Train CSV": TRAIN_CSV_PATH,
    "Train images": TRAIN_IMAGE_DIR,
    "Validation CSV": VAL_CSV_PATH,
    "Validation images": VAL_IMAGE_DIR,
    "Test CSV": TEST_CSV_PATH,
    "Test images": TEST_IMAGE_DIR,
}

for name, path in paths_to_check.items():
    print(
        f"{name}: {path.exists()} | {path}"
    )

train_df = pd.read_csv(
    TRAIN_CSV_PATH
)

val_df = pd.read_csv(
    VAL_CSV_PATH
)

test_df = pd.read_csv(
    TEST_CSV_PATH
)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nValidation columns:")
print(val_df.columns.tolist())

print("\nTest columns:")
print(test_df.columns.tolist())

from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
)


def normalize_boneage_dataframe(dataframe):
    dataframe = dataframe.copy()

    # 컬럼 이름 정리
    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            r"[^a-z0-9]+",
            "_",
            regex=True,
        )
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

    dataframe = dataframe.rename(
        columns=rename_map
    )

    required_columns = {
        "id",
        "boneage",
        "male",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"필수 컬럼 누락: {missing_columns}"
        )

    # ID
    dataframe["id"] = (
        dataframe["id"]
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    # 골연령
    dataframe["boneage"] = pd.to_numeric(
        dataframe["boneage"],
        errors="raise",
    ).astype(float)

    # 성별
    original_male = dataframe["male"]

    if is_bool_dtype(original_male):
        dataframe["male"] = (
            original_male.astype(float)
        )

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

        dataframe["male"] = (
            normalized_male.map({
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
        )

    if dataframe["male"].isna().any():
        failed_values = (
            original_male[
                dataframe["male"].isna()
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"성별 변환 실패 값: {failed_values}"
        )

    invalid_values = (
        set(dataframe["male"].unique())
        - {0.0, 1.0}
    )

    if invalid_values:
        raise ValueError(
            "성별에 0/1 이외의 값이 있습니다: "
            f"{invalid_values}"
        )

    return dataframe[
        ["id", "boneage", "male"]
    ].reset_index(drop=True)


train_df = normalize_boneage_dataframe(
    train_df
)

val_df = normalize_boneage_dataframe(
    val_df
)

test_df = normalize_boneage_dataframe(
    test_df
)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

train_ids = set(train_df["id"])
val_ids = set(val_df["id"])
test_ids = set(test_df["id"])

print(
    "Train-Validation ID 중복:",
    len(train_ids & val_ids),
)

print(
    "Train-Test ID 중복:",
    len(train_ids & test_ids),
)

print(
    "Validation-Test ID 중복:",
    len(val_ids & test_ids),
)



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
            and image_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):
            image_id = image_path.stem

            if image_id in image_index:
                raise ValueError(
                    f"중복 이미지 ID 발견: {image_id}"
                )

            image_index[image_id] = str(
                image_path
            )

    print(
        f"{image_dir.name}: "
        f"{len(image_index)}개 검색"
    )

    return image_index


train_image_index = build_image_index(
    TRAIN_IMAGE_DIR
)

val_image_index = build_image_index(
    VAL_IMAGE_DIR
)

test_image_index = build_image_index(
    TEST_IMAGE_DIR
)



def attach_image_paths(
    dataframe,
    image_index,
    dataset_name,
):
    dataframe = dataframe.copy()

    dataframe["image_path"] = (
        dataframe["id"].map(
            image_index
        )
    )

    missing_df = dataframe[
        dataframe["image_path"].isna()
    ]

    print(
        f"{dataset_name} 이미지 누락:",
        len(missing_df),
    )

    if len(missing_df) > 0:
        print(
            "누락 ID 예시:",
            missing_df["id"]
            .head(10)
            .tolist(),
        )

        raise FileNotFoundError(
            f"{dataset_name} 이미지 "
            f"{len(missing_df)}개 누락"
        )

    return dataframe



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

test_df = attach_image_paths(
    test_df,
    test_image_index,
    "Test",
)



def print_dataset_summary(
    name,
    dataframe,
):
    print(f"\n[{name}]")
    print("개수:", len(dataframe))
    print(
        "골연령 평균:",
        round(
            dataframe["boneage"].mean(),
            3,
        ),
    )
    print(
        "골연령 표준편차:",
        round(
            dataframe["boneage"].std(),
            3,
        ),
    )
    print(
        "최솟값:",
        dataframe["boneage"].min(),
    )
    print(
        "최댓값:",
        dataframe["boneage"].max(),
    )
    print(
        "남아 비율:",
        round(
            dataframe["male"].mean(),
            3,
        ),
    )


print_dataset_summary(
    "Train",
    train_df,
)

print_dataset_summary(
    "Validation",
    val_df,
)

print_dataset_summary(
    "Test",
    test_df,
)


train_transform = transforms.Compose([
    transforms.Resize(
        (299, 299),
        interpolation=InterpolationMode.BICUBIC,
    ),

    transforms.RandomAffine(
        degrees=5,
        translate=(0.03, 0.03),
        scale=(0.97, 1.03),
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


val_transform = transforms.Compose([
    transforms.Resize(
        (299, 299),
        interpolation=InterpolationMode.BICUBIC,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


class BoneAgeDataset(Dataset):
    def __init__(
        self,
        dataframe,
        transform,
    ):
        self.dataframe = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        with Image.open(
            row["image_path"]
        ) as image:
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
    

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_WORKERS = 0


train_dataset = BoneAgeDataset(
    dataframe=train_df,
    transform=train_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


val_dataset = BoneAgeDataset(
    dataframe=val_df,
    transform=val_transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)



sample_batch = next(
    iter(train_loader)
)

print(
    "Image shape:",
    sample_batch["image"].shape,
)

print(
    "Sex shape:",
    sample_batch["sex"].shape,
)

print(
    "Bone age shape:",
    sample_batch["bone_age"].shape,
)

print(
    "Bone age examples:",
    sample_batch["bone_age"][:5],
)


class XceptionLinearBoneAgeModel(nn.Module):
    def __init__(
        self,
        pretrained=True,
    ):
        super().__init__()

        # ImageNet 사전학습 Xception
        # 기존 classification head는 사용하지 않음
        self.backbone = timm.create_model(
            "legacy_xception",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # Xception feature map
        # [B, 2048, 10, 10]
        # → [B, 256, 10, 10]
        # → [B, 256, 3, 3]
        # → [B, 2304]
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

        # 성별 1차원 → 32차원
        self.sex_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
        )

        # 2304 image + 32 gender
        # → 단일 선형 회귀
        self.regressor = nn.Linear(
            2304 + 32,
            1,
        )

    def forward(
        self,
        image,
        sex,
    ):
        feature_map = (
            self.backbone.forward_features(
                image
            )
        )

        image_feature = self.image_head(
            feature_map
        )

        sex_feature = self.sex_embedding(
            sex
        )

        combined_feature = torch.cat(
            [
                image_feature,
                sex_feature,
            ],
            dim=1,
        )

        prediction = (
            self.regressor(
                combined_feature
            )
            .squeeze(1)
        )

        return prediction
    


gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

model = XceptionLinearBoneAgeModel(
    pretrained=True
).to(device)

# 처음부터 전체 End-to-End 학습
for parameter in model.parameters():
    parameter.requires_grad = True

print(
    "Model device:",
    next(model.parameters()).device,
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(
    "Total parameters:",
    f"{total_parameters:,}",
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}",
)



model.eval()

test_images = (
    sample_batch["image"][:2]
    .to(device)
)

test_sexes = (
    sample_batch["sex"][:2]
    .to(device)
)

with torch.no_grad():
    feature_map = (
        model.backbone.forward_features(
            test_images
        )
    )

    conv_feature = model.image_head[0](
        feature_map
    )

    activated_feature = model.image_head[1](
        conv_feature
    )

    pooled_feature = model.image_head[2](
        activated_feature
    )

    image_vector = model.image_head[3](
        pooled_feature
    )

    sex_vector = model.sex_embedding(
        test_sexes
    )

    predictions = model(
        test_images,
        test_sexes,
    )

print(
    "Xception feature:",
    feature_map.shape,
)

print(
    "Conv feature:",
    conv_feature.shape,
)

print(
    "Pool feature:",
    pooled_feature.shape,
)

print(
    "Image vector:",
    image_vector.shape,
)

print(
    "Sex vector:",
    sex_vector.shape,
)

print(
    "Prediction:",
    predictions.shape,
)




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

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            images,
            sexes,
        )

        loss = criterion(
            predictions,
            targets,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = targets.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_absolute_error += (
            torch.abs(
                predictions.detach()
                - targets
            )
            .sum()
            .item()
        )

        total_samples += batch_size

    return {
        "loss":
            total_loss / total_samples,

        "mae":
            total_absolute_error
            / total_samples,
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

            predictions = model(
                images,
                sexes,
            )

            loss = criterion(
                predictions,
                targets,
            )

            errors = predictions - targets
            batch_size = targets.size(0)

            total_loss += (
                loss.item() * batch_size
            )

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
                predictions
                .cpu()
                .numpy()
            )

            all_targets.extend(
                targets
                .cpu()
                .numpy()
            )

            all_ids.extend(
                batch["id"]
            )

    return {
        "loss":
            total_loss / total_samples,

        "mae":
            total_absolute_error
            / total_samples,

        "rmse":
            np.sqrt(
                total_squared_error
                / total_samples
            ),

        "predictions":
            np.asarray(
                all_predictions
            ),

        "targets":
            np.asarray(
                all_targets
            ),

        "ids":
            all_ids,
    }




criterion = nn.L1Loss()

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.backbone.parameters(),
            "lr": 1e-4,
        },
        {
            "params": model.image_head.parameters(),
            "lr": 1e-3,
        },
        {
            "params": model.sex_embedding.parameters(),
            "lr": 1e-3,
        },
        {
            "params": model.regressor.parameters(),
            "lr": 1e-3,
        },
    ],
    weight_decay=1e-4,
)



scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    min_lr=1e-7,
)


IMAGE_SIZE = 299




MAX_EPOCHS = 35
EARLY_STOPPING_PATIENCE = 8

best_val_mae = float("inf")
no_improvement_count = 0
history = []

for epoch in range(
    1,
    MAX_EPOCHS + 1,
):
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

    scheduler.step(
        val_result["mae"]
    )

    backbone_lr = (
        optimizer.param_groups[0]["lr"]
    )

    head_lr = (
        optimizer.param_groups[1]["lr"]
    )

    history.append({
        "epoch": epoch,
        "train_loss":
            train_result["loss"],
        "train_mae":
            train_result["mae"],
        "val_loss":
            val_result["loss"],
        "val_mae":
            val_result["mae"],
        "val_rmse":
            val_result["rmse"],
        "backbone_lr":
            backbone_lr,
        "head_lr":
            head_lr,
    })

    print(
        f"Epoch {epoch:02d} | "
        f"Train MAE: "
        f"{train_result['mae']:.3f}개월 | "
        f"Val MAE: "
        f"{val_result['mae']:.3f}개월 | "
        f"Val RMSE: "
        f"{val_result['rmse']:.3f}개월 | "
        f"Backbone LR: "
        f"{backbone_lr:.2e} | "
        f"Head LR: "
        f"{head_lr:.2e}"
    )

    if val_result["mae"] < best_val_mae:
        best_val_mae = val_result["mae"]
        no_improvement_count = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mae": best_val_mae,
                "history": history,
                "model_name": "XceptionLinearBoneAgeModel",
                "backbone_lr": 3e-5,
                "head_lr": 1e-4,
                "image_size": IMAGE_SIZE,
                "train_batch_size": TRAIN_BATCH_SIZE,
            },
            BEST_MODEL_PATH,
        )

        print(
            "최적 모델 저장:",
            f"{best_val_mae:.3f}개월",
        )

    else:
        no_improvement_count += 1

        print(
            "성능 미개선:",
            f"{no_improvement_count}/"
            f"{EARLY_STOPPING_PATIENCE}",
        )

        pd.DataFrame(history).to_csv(
        HISTORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if (
        no_improvement_count
        >= EARLY_STOPPING_PATIENCE
    ):
        print("Early stopping")
        break