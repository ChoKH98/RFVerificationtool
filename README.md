# RFVerificationtool

RF/마이크로파 회로 및 PCB EM 검증 툴

이 저장소는 3D Yee-grid FDTD 기반으로 2.4 GHz 마이크로스트립 패치 안테나의
기하 생성, 가우시안 포트 excitation, S11 추출을 검증하는 프로토타입입니다.

## 구조

```
RFVerificationtool/
├── em_solver/          — FDTD + CPML EM 시뮬레이터
│   ├── fdtd_core.py    — 3D Yee 격자 시간도메인 솔버
│   ├── cpml.py         — Convolutional PML 흡수 경계
│   ├── geometry.py     — PCB 레이어 스택 + 패치 기하
│   ├── excitation.py   — 가우시안 포트 + S파라미터 추출
│   └── run_patch_sim.py — 2.4 GHz 패치 검증 실행
├── tests/              — 빠른 회귀 테스트
├── docs/               — 알고리즘 문서
├── results/            — 생성된 시뮬레이션 결과 안내
└── examples/           — 사용 예제
```

## 목표 정확도
- 단층 패치 S11 공진: ±50 MHz 이내
- 다층 PCB 층간 커플링: ±1 dB
- 비아 임피던스: ±5%

## 기술 기반
- FDTD: Yee (1966) 격자 이산화
- CPML: Roden & Gedney (2000) 흡수 경계
- 재료: Rogers 4003C (εr=3.55, tanδ=0.0027)
- 검증 기준: OpenEMS + 해석적 공식

## 사용법

의존성:

```bash
python -m pip install -r requirements.txt
```

기존 로컬 가상환경을 사용하는 경우:

```bash
cd /home/whqkrel/RFVerificationtool
source /home/whqkrel/rfic_project/venv/bin/activate
python em_solver/run_patch_sim.py
```

빠른 smoke 실행:

```bash
python em_solver/run_patch_sim.py --steps 20 --quiet --points 21 --output-dir results/_tmp_smoke
```

테스트:

```bash
python -m unittest discover -s tests
```

## 현재 한계

- CPML 보정항은 현재 y-normal face에 우선 구현되어 있으며 x/z face 보정은 확장 대상입니다.
- OpenEMS 대조 검증 파일은 아직 저장소에 포함되어 있지 않습니다.
- 결과 CSV/PNG는 재생성 산출물이므로 기본적으로 Git 추적에서 제외합니다.
