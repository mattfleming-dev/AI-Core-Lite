#ifndef PayloadDir
  #error PayloadDir is required
#endif
#ifndef InstallerOutputDir
  #define InstallerOutputDir "..\dist"
#endif

[Setup]
AppId={{6D0C4B7E-2F1A-4C58-9B1E-3A7D5E8F1C42}
AppName=Lite RAG
AppVersion=0.1.0
AppPublisher=Local AI Platform
DefaultDirName={localappdata}\Lite-RAG
DefaultGroupName=Lite RAG
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupArchitecture=x64
MinVersion=10.0.17763
OutputDir={#InstallerOutputDir}
OutputBaseFilename=Lite-RAG-0.1.0-Windows-x64-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\runtime\pythonw.exe
CloseApplications=no

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"
Name: autostart; Description: "Start Lite RAG in the background when I sign in"; Flags: unchecked

[Files]
Source: "{#PayloadDir}\runtime\*"; DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#PayloadDir}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#PayloadDir}\config\config.json"; DestDir: "{app}\config"; Flags: onlyifdoesntexist uninsneveruninstall
Source: "{#PayloadDir}\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Dirs]
Name: "{app}\data"; Flags: uninsneveruninstall

[Icons]
Name: "{group}\Lite RAG"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-m lite_rag --config ""{app}\config\config.json"" app"; WorkingDir: "{app}\app"
Name: "{group}\Stop Lite RAG"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-m lite_rag --config ""{app}\config\config.json"" stop"; WorkingDir: "{app}\app"
Name: "{group}\Edit settings file"; Filename: "{app}\config\config.json"
Name: "{group}\Readme"; Filename: "{app}\README.md"
Name: "{autodesktop}\Lite RAG"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-m lite_rag --config ""{app}\config\config.json"" app"; WorkingDir: "{app}\app"; Tasks: desktopicon
Name: "{userstartup}\Lite RAG"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-m lite_rag --config ""{app}\config\config.json"" app --no-browser"; WorkingDir: "{app}\app"; Tasks: autostart

[Run]
Filename: "{app}\runtime\pythonw.exe"; Parameters: "-m lite_rag --config ""{app}\config\config.json"" app"; WorkingDir: "{app}\app"; Description: "Open Lite RAG"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\runtime\pythonw.exe"; Parameters: "-m lite_rag --config ""{app}\config\config.json"" stop"; WorkingDir: "{app}\app"; Flags: runhidden; RunOnceId: "StopLiteRag"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\app"
