{
  inputs.flakelight.url = "github:nix-community/flakelight";
  outputs = {flakelight, ...}:
    flakelight ./. {
      systems = ["aarch64-darwin" "x86_64-linux"];
      flakelight.builtinFormatters = false;
      devShell = {
        packages = pkgs: let
          nixTooling = with pkgs; [nix alejandra];
          pythonTooling = with pkgs; [python311 uv ruff ty];
          matplotlibTex = pkgs.texlive.combine {
            inherit
              (pkgs.texlive)
              scheme-basic
              amsmath
              latexmk
              cm-super
              type1cm
              xcolor
              underscore
              dvipng
              ;
          };
        in
          nixTooling ++ pythonTooling ++ [matplotlibTex];
        shellHook = ''
          source .venv/bin/activate
        '';
      };
    };
}
