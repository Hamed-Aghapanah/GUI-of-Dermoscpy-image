# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)
hiddenimport=["pywt","pywt._estentions._cwt", "sklearn.neighbors.typedefs"]


block_cipher = None


a = Analysis(['dermatology.py'],
             pathex=['D:\\001phd\\2OpenCV\\project1\\final'],
             binaries=[],
             datas=[],
             hiddenimports=['pywt', 'pywt._extensions._cwt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='dermatology',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False )
