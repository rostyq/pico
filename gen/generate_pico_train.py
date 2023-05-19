from dataclasses import dataclass
from struct import pack
from typing import Iterable, BinaryIO, Generator, Optional
from pathlib import Path
from contextlib import suppress
from io import BytesIO

from cv2 import Mat


@dataclass
class Target:
    x: float
    y: float
    s: float

    def scale(self, s: float) -> "Target":
        return self.__class__(self.x * s, self.y * s, self.s * s)

    def __bytes__(self) -> bytes:
        return pack("iii", int(self.y), int(self.x), int(self.s))

    def draw(
        self, img: Mat, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 1
    ):
        from cv2 import circle

        circle(img, (int(self.x), int(self.y)), int(self.s / 2), color, thickness)

    @classmethod
    def from_csv(
        cls, csvfile: Iterable[str], output: dict[str, list["Target"]], **kwargs
    ):
        from csv import reader

        for row in reader(csvfile, **kwargs):
            image_name, x, y, size = row

            with suppress(ValueError):
                targets = output.setdefault(image_name, [])
                targets.append(Target(float(x), float(y), float(size)))


def serialize_image(img: Mat) -> bytes:
    assert img.ndim == 2
    assert img.dtype == "uint8"

    h, w = img.shape
    size = h * w

    return pack("ii", h, w) + pack(f"{size}B", *img.reshape(-1))


def write_sample(img: Mat, ts: list[Target], dst: BinaryIO):
    dst.write(serialize_image(img))
    dst.write(pack("i", len(ts)))
    dst.write(b"".join([bytes(t) for t in ts]))

    dst.flush()


def generate_scales(
    n: int, lower: float = 0.7, upper: float = 1.3
) -> Generator[float, None, None]:
    from random import uniform

    for _ in range(n):
        yield uniform(lower, upper)


def draw_flip() -> bool:
    from random import getrandbits

    return bool(getrandbits(1))


def export_sample(
    img: Mat,
    ts: list[Target],
    dst: BinaryIO,
    perturbs: int = 8,
    min_size: int = 24,
    scale: tuple[float, float] = (0.7, 1.3),
):
    from cv2 import resize, flip

    for scale in generate_scales(n=perturbs, lower=scale[0], upper=scale[1]):
        resized = resize(img, (0, 0), fx=scale, fy=scale)

        to_flip = draw_flip()

        if to_flip:
            resized = flip(resized, 1)

        targets = [t.scale(scale) for t in ts]
        targets = [t for t in targets if t.s >= min_size]

        if to_flip:
            w = resized.shape[1]
            targets = [Target(w - t.x, t.y, t.s) for t in targets]

        write_sample(resized, targets, dst)


def process_image(path: Path, ts: list[Target], dst: BinaryIO):
    from cv2 import imread, cvtColor, COLOR_BGR2GRAY, COLOR_BGRA2GRAY

    img = imread(str(path))

    if img.ndim == 3:
        img = cvtColor(img, COLOR_BGR2GRAY)
    elif img.ndim == 4:
        img = cvtColor(img, COLOR_BGRA2GRAY)

    export_sample(img, ts, dst)


def load_image_targets(path: Path) -> dict[str, list[Target]]:
    image_targets: dict[str, list[Target]] = {}

    with path.open("r") as f:
        Target.from_csv(f, image_targets)

    return image_targets


def scan_for_images(path: Path) -> Generator[Path, None, None]:
    from itertools import chain

    return chain(path.glob("*.jpg"), path.glob("*.png"))


def buffer_process_image(path: Path, targets: list[Target]) -> bytes:
    with BytesIO(b"") as buffer:
        process_image(path, targets, buffer)
        return buffer.getvalue()


def handle_item(item: tuple[Path, list[Target]]) -> bytes:
    return buffer_process_image(item[0], item[1])


def main(
    images: Path,
    output: Path = Path("./pico-train.bin"),
    annotations: str = "pico-targets.csv",
    max_workers: Optional[int] = None,
):
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    annots = load_image_targets(images.joinpath(annotations))
    paths = list(scan_for_images(images))
    count = len(paths)

    def iter_images(
        paths: list[Path], annots: dict[str, list[Target]]
    ) -> Generator[tuple[Path, list[Target]], None, None]:
        for path in paths:
            yield (path, annots.get(path.name, []))

    with open(output, "wb+") as buffer:
        # for path, targets in tqdm(iter_images(paths, annots), total=count):
        #     process_image(path, targets, output)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for data in tqdm(
                executor.map(handle_item, iter_images(paths, annots), chunksize=5),
                total=count,
            ):
                buffer.write(data)
                buffer.flush()


if __name__ == "__main__":
    from typer import run

    run(main)
