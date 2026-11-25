; Inno Setup script for the step-by-step Windows installer (built by packaging/build.py with iscc).
;   iscc /DAppVersion=1.1.1 /DSourceDir=<dist\Noelle> /DOutputDir=<dist> packaging\windows\noelle.iss
;
; Pages: Welcome -> License -> Install mode (just me / all users) -> Location -> Start menu ->
;        Additional tasks -> Hugging Face account (optional token) -> Ready -> Install -> Finish.
; The token is handed to Noelle.exe through a file in Setup's private temp folder, imported into
; Windows Credential Manager as the user who started Setup, and deleted immediately.

#ifndef AppVersion
  #define AppVersion "1.1.1"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\Noelle"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

[Setup]
AppId={{8C1D7E2A-5B3F-4E8B-9A6C-2F4D1B7E9C30}
AppName=Noelle AI Studio
AppVersion={#AppVersion}
AppVerName=Noelle AI Studio {#AppVersion}
AppPublisher=Noelle
DefaultDirName={autopf}\Noelle
DefaultGroupName=Noelle
DisableProgramGroupPage=auto
LicenseFile=..\..\LICENSE
OutputDir={#OutputDir}
OutputBaseFilename=Noelle-{#AppVersion}-windows-x64-setup
SetupIconFile=..\noelle.ico
UninstallDisplayIcon={app}\Noelle.exe
WizardStyle=modern
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Noelle AI Studio"; Filename: "{app}\Noelle.exe"
Name: "{group}\{cm:UninstallProgram,Noelle AI Studio}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Noelle AI Studio"; Filename: "{app}\Noelle.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Noelle.exe"; Parameters: "setup"; Description: "Open Noelle and finish setup"; \
  Flags: nowait postinstall skipifsilent runasoriginaluser

[Code]
var
  TokenPage: TInputQueryWizardPage;

procedure InitializeWizard;
begin
  TokenPage := CreateInputQueryPage(wpSelectTasks,
    'Hugging Face account', 'Optional: connect your Hugging Face account',
    'Paste a Hugging Face access token to download gated or private models. It is kept locally in ' +
    'Windows Credential Manager, never in a file, and never transferred to any third party other than Hugging Face. ' +
    'Leave it empty to skip; you can add it later in Noelle''s Settings.');
  TokenPage.Add('Access token (hf_...):', True);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Token: String;
begin
  Result := True;
  if CurPageID = TokenPage.ID then
  begin
    Token := Trim(TokenPage.Values[0]);
    if (Token <> '') and (Pos('hf_', Token) <> 1) then
      Result := MsgBox('This does not look like a Hugging Face token (they start with "hf_"). Use it anyway?',
                       mbConfirmation, MB_YESNO) = IDYES;
  end;
end;

function TryTokenHandoff(const Token, TokenFile: String): Boolean;
var
  ResultCode: Integer;
begin
  Result := False;
  ResultCode := -1;
  if not SaveStringToFile(TokenFile, Token, False) then
  begin
    Log('Could not write the token hand-off file: ' + TokenFile);
    exit;
  end;
  try
    Result := ExecAsOriginalUser(ExpandConstant('{app}\Noelle.exe'),
                'setup --import-hf-token "' + TokenFile + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode)
              and (ResultCode = 0);
    Log(Format('Hugging Face token import from %s exited with code %d', [TokenFile, ResultCode]));
  finally
    DeleteFile(TokenFile);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Token, TokenFile: String;
  Saved: Boolean;
begin
  if CurStep = ssPostInstall then
  begin
    Token := Trim(TokenPage.Values[0]);
    if Token <> '' then
    begin
      if IsAdminInstallMode then
      begin
        // Elevation can make Setup's {tmp} unreadable to the original user. Use that user's
        // roaming profile instead; Noelle deletes it immediately after reading it.
        TokenFile := ExpandConstant('{userappdata}\Noelle\pending-hf-token-' +
                                     GetDateTimeString('yyyymmddhhnnss', #0, #0) + '.txt');
        ForceDirectories(ExtractFileDir(TokenFile));
      end
      else
        TokenFile := ExpandConstant('{tmp}\hf_token.txt');  // private per-user temp, auto-cleaned by Setup
      Saved := TryTokenHandoff(Token, TokenFile);
      if not Saved then
      begin
        // Fall back to the other location for unusual profiles or locked-down ACLs.
        if IsAdminInstallMode then
          TokenFile := ExpandConstant('{tmp}\hf_token.txt')
        else
        begin
          TokenFile := ExpandConstant('{userappdata}\Noelle\pending-hf-token-' +
                                       GetDateTimeString('yyyymmddhhnnss', #0, #0) + '.txt');
          ForceDirectories(ExtractFileDir(TokenFile));
        end;
        Saved := TryTokenHandoff(Token, TokenFile);
      end;
      if not Saved then
        MsgBox('Noelle could not save the Hugging Face token. Open Noelle and paste it in ' +
               'Settings › Account, or run the installer again.', mbInformation, MB_OK);
    end;
  end;
end;
