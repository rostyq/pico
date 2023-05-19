from dataclasses import dataclass
from typing import Iterable, Optional
from pathlib import Path
from csv import reader, writer
from math import hypot


@dataclass
class Annotation:
    image_name: str

    leye_x: float  # 0
    leye_y: float  # 1

    reye_x: float  # 2
    reye_y: float  # 3

    nose_x: float  # 4
    nose_y: float  # 5

    mouth_x: float  # 6
    mouth_y: float  # 7

    def target(self) -> tuple[float, float, float]:
        # eyedist = ( (face[0]-face[2])**2 + (face[1]-face[3])**2 )**0.5
        d = hypot(self.leye_x - self.reye_x, self.leye_y - self.reye_y)

        # r = (face[1]+face[3])/2.0 + 0.25*eyedist
        y = (self.leye_y + self.reye_y) / 2.0 + 0.25 * d

        # c = (face[0]+face[2])/2.0
        x = (self.leye_x + self.reye_x) / 2.0

        # s = 2.0*1.5*eyedist
        s = 2.0 * 1.5 * d

        return (x, y, s)


def parse_annotations(csvfile: Iterable[str]) -> list[Annotation]:
    result: list[Annotation] = []

    for row in reader(csvfile, delimiter=" "):
        [image_name, *data, _] = row
        data = [float(value) for value in data]

        result.append(Annotation(image_name, *data))

    return result


def main(annotation_path: Path, output_path: Optional[Path] = None):
    with annotation_path.open("r") as f:
        annots = parse_annotations(f)

    if output_path is None:
        output_path = annotation_path.parent / "pico-targets.csv"

    with output_path.open("w") as output:
        w = writer(output, lineterminator="\n")
        w.writerow(["image-name", "target-x", "target-y", "target-size"])

        for annot in annots:
            (target_x, target_y, target_size) = annot.target()
            
            target_x = round(target_x, 1)
            target_y = round(target_y, 1)
            target_size = round(target_size, 1)

            w.writerow([annot.image_name, target_x, target_y, target_size])


if __name__ == "__main__":
    from typer import run

    run(main)
