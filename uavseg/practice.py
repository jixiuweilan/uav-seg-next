"""Generate a portable, synthetic-only review exercise; no external data inputs."""

import argparse
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw

from .audit import audit_data
from .common import DataPaths, canonical, sha256
from .review import review_page
from .scenes import extract_features, screen_features


def build_practice(output):
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        for name in ('train/images', 'train/masks', 'test/images'):
            (base / 'raw' / name).mkdir(parents=True)
        image = Image.new('RGB', (1024, 1024), '#b3caa0')
        draw = ImageDraw.Draw(image)
        draw.line([(0, 620), (450, 420), (1024, 480)], fill='#ddd4bd', width=85)
        draw.line([(400, 0), (450, 420), (650, 1024)], fill='#ddd4bd', width=55)
        draw.rectangle((80, 70, 250, 210), fill='#914c36')
        draw.rectangle((600, 120, 840, 230), fill='#82462c')
        draw.ellipse((70, 700, 300, 980), fill='#4c90ac')
        other = Image.new('RGB', (1024, 1024), '#b3caa0')
        draw = ImageDraw.Draw(other)
        draw.line([(0, 350), (1024, 350)], fill='#ddd4bd', width=85)
        draw.rectangle((600, 650, 920, 840), fill='#914c36')
        draw.ellipse((40, 50, 320, 280), fill='#4c90ac')
        samples = {'practice-a': image, 'practice-b': image.transpose(Image.Transpose.ROTATE_90),
                   'practice-c': other, 'practice-d': Image.new('RGB', (1024, 1024), '#b3caa0')}
        for name, sample in samples.items():
            sample.save(base / 'raw/train/images' / (name + '.png'))
            Image.new('L', (1024, 1024), 1).save(base / 'raw/train/masks' / (name + '.png'))
        Image.new('RGB', (1024, 1024)).save(base / 'raw/test/images/practice-test.png')
        config = base / 'paths.json'
        config.write_bytes(canonical({'dataset_root': 'raw', 'access': 'read-only',
                                     'train_images': 'train/images', 'train_masks': 'train/masks',
                                     'test_images': 'test/images'}))
        paths = DataPaths(config)
        document = audit_data(paths)
        document['manifest']['purpose'] = 'synthetic-review-practice'
        document['manifest_sha256'] = sha256(canonical(document['manifest']))
        screen = screen_features(extract_features(paths, document), document,
                                 {'max_per_partition': 1, 'controls': 1})
        result = review_page(paths, document, screen, output, [base], practice=True)
        return document, screen, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成无需安装环境即可打开的中文合成练习页（开发端使用）。')
    parser.add_argument('--output', type=Path, required=True, help='新的 .local/ HTML 文件路径')
    args = parser.parse_args()
    _, _, result = build_practice(args.output)
    print(f"已生成中文练习页：{args.output}；练习关系 {result['review_pairs']} 条。")
