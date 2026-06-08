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
│   ├── compare_s11.py  — HFSS/openEMS S11 비교 유틸리티
│   ├── benchmark.py    — 다중 구조 벤치마크 비교 CLI
│   ├── validation.py   — S-parameter/field/RCS 표준 포맷 로더 + 메트릭
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

HFSS/openEMS S11 비교:

```bash
python -m em_solver.compare_s11 results/patch_s11.csv references/openems_patch_s11.csv --label openEMS
python -m em_solver.compare_s11 results/patch_s11.csv references/hfss_patch_s11.csv --label HFSS
```

다중 구조 HFSS/openEMS 벤치마크:

```bash
python -m em_solver.external_solvers status
python -m em_solver.benchmark list-cases
python -m em_solver.bridge --root /mnt/c/Users/whqkr/Desktop/RFVerificationBridge
python -m em_solver.external_solvers generate-openems --case horn_xband --output-dir references/horn_xband/openems_runner
python -m em_solver.benchmark compare \
  --case horn_xband \
  --candidate-dir results/horn_xband/fdtd \
  --hfss-dir references/horn_xband/hfss \
  --openems-dir references/horn_xband/openems \
  --output-dir results/horn_xband/benchmark
```

표준 artifact 이름:

- `sparams.csv`: `frequency_hz,port_i,port_j,real,imag,db,phase_deg`
- `field_near.npz`, `field_medium.npz`: `frequency_hz`, `points_m`, complex `Ex/Ey/Ez/Hx/Hy/Hz`
- `field_far.npz`: `frequency_hz`, `theta_deg`, `phi_deg`, complex `Etheta/Ephi`, optional `gain_dbi`
- `rcs.csv`: `frequency_hz,theta_deg,phi_deg,rcs_dbsm`

벤치마크 출력:

- `metrics.json`: pass/fail 포함 전체 메트릭
- `summary.csv`: CI/스크립트용 flat metric table
- `optimization_recommendations.json`
- `optimization_recommendations.csv`

Windows 바탕화면 브리지:

```text
C:\Users\whqkr\Desktop\RFVerificationBridge
```

Windows PowerShell에서 HFSS/PyAEDT 상태 확인:

```powershell
cd "$env:USERPROFILE\Desktop\RFVerificationBridge\runners\hfss"
py -m pip install ansys-aedt-core
py .\hfss_status.py
```

첫 HFSS 기준 데이터 생성(`waveguide_family`):

```powershell
cd "$env:USERPROFILE\Desktop\RFVerificationBridge\runners\hfss"
py .\run_hfss_waveguide_family.py --non-graphical
```

전체 브리지 상태/비교 루프:

```bash
python -m em_solver.workflow --bridge-root /mnt/c/Users/whqkr/Desktop/RFVerificationBridge setup
python -m em_solver.workflow --bridge-root /mnt/c/Users/whqkr/Desktop/RFVerificationBridge status
python -m em_solver.workflow --bridge-root /mnt/c/Users/whqkr/Desktop/RFVerificationBridge generate-candidate --case waveguide_family
python -m em_solver.workflow --bridge-root /mnt/c/Users/whqkr/Desktop/RFVerificationBridge run --case waveguide_family
```

## 현재 한계

- CPML 보정항은 x/y/z face에 구현되어 있으나 평면파 반사율 테스트와 외부 솔버 상관 검증이 필요합니다.
- HFSS/openEMS 대조 검증 파일은 아직 저장소에 포함되어 있지 않으며, v1은 표준 CSV/NPZ 결과 입력 비교 방식입니다.
- phased array/horn/waveguide/corner scattering 케이스는 벤치마크 하네스와 메트릭이 구현되어 있습니다.
- 현재 환경에서는 openEMS 바이너리는 있으나 Octave/MATLAB 런타임이 없고 HFSS 실행기(`ansysedt`)가 PATH에 없어 외부 solver를 직접 실행할 수 없습니다.
- 결과 CSV/PNG는 재생성 산출물이므로 기본적으로 Git 추적에서 제외합니다.
