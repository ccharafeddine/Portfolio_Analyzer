; Inno Setup script for Portfolio Analyzer.
; Compiled in CI from the repo root after PyInstaller produces dist\PortfolioAnalyzer\.
;   ISCC /DAppVersion=1.0.1 packaging\installer.iss   ->  PortfolioAnalyzer-Setup.exe

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "Portfolio Analyzer"
#define AppExe "PortfolioAnalyzer.exe"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=ccharafeddine
DefaultDirName={autopf}\Portfolio Analyzer
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExe}
OutputDir=.
OutputBaseFilename=PortfolioAnalyzer-Setup
SetupIconFile=src\ui\assets\app.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
; Resolve relative paths below from the repo root (this .iss lives in packaging\).
SourceDir={#SourcePath}\..

[Files]
Source: "dist\PortfolioAnalyzer\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\Portfolio Analyzer"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\Portfolio Analyzer"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch Portfolio Analyzer"; Flags: nowait postinstall skipifsilent
