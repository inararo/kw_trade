# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller 패키징 설정 파일 (.spec)
- PyQt6, Stable-Baselines3 (PyTorch), Numba 등 무거운 라이브러리를 포함한 프로젝트 패키징.
- 명령어: pyinstaller kiwoom_scalping_quant.spec
"""

block_cipher = None

# Stable-Baselines3와 PyTorch는 동적 로딩이 많아 hiddenimports에 명시적으로 추가해야 의존성 누락 에러(ModuleNotFoundError)를 방지할 수 있습니다.
hidden_imports = [
    'PyQt6',
    'qasync',
    'stable_baselines3',
    'sb3_contrib',
    'torch',
    'gymnasium',
    'numpy',
    'numba',
    'dependency_injector',
    'returns',
    'websockets'
]

a = Analysis(
    ['main.py'],
    pathex=['.'], # 프로젝트 루트 경로
    binaries=[],
    datas=[
        # (원본 경로, 패키징 내부 경로)
        ('config.yaml', '.'),
        ('.env', '.') # 주의: 실 배포 시 .env는 제외하고 환경 변수로 주입하는 것을 권장. 테스트용으로 포함.
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', # 사용하지 않는 무거운 라이브러리 제외하여 빌드 용량 최적화
        'tensorboard'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 단일 실행 파일(Single File, --onefile) 방식
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='KiwoomScalpingQuant', # 최종 생성될 실행 파일 이름
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,      # UPX 압축 활성화 (설치되어 있어야 함)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, # Windows: False면 터미널 창 없이 GUI만 뜸. macOS에선 큰 의미 없음.
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icon.ico' # 아이콘 경로 추가 가능
)

# macOS 앱 번들(.app) 전용 설정 (Windows에서 빌드할 경우 무시됨)
app = BUNDLE(
    exe,
    name='KiwoomScalpingQuant.app',
    icon=None,
    bundle_identifier='com.quant.kiwoomscalping',
    info_plist={
        'NSHighResolutionCapable': 'True'
    }
)
