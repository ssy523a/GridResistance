# Windows portable distribution

Build on 64-bit Windows from the project root:

```powershell
.\build_distribution.ps1
```

If the local PowerShell execution policy blocks scripts, use this command
without changing the system-wide policy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build_distribution.ps1
```

The build produces:

- `dist\GridResistanceAnalyzer\GridResistanceAnalyzer.exe`
- `dist\GridResistanceAnalyzer-windows-x64.zip`

Distribute the ZIP file. Users must extract the entire directory before
starting `GridResistanceAnalyzer.exe`; the executable cannot be separated
from its `_internal` directory.

The application does not require a separate Python installation. Material
library changes are stored in the current user's application-data directory.

Before public distribution, review the licenses of bundled dependencies,
especially Gmsh (GPL), and consider signing the executable to reduce Windows
SmartScreen warnings.
