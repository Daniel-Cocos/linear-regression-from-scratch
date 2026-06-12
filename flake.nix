{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: let
    pkgs = nixpkgs.legacyPackages.x86_64-linux;

    # Python 3.11 with sphinx pinned to 8.x
    python = pkgs.python311.override {
      packageOverrides = self: super: {
        sphinx = super.sphinx.overridePythonAttrs (old: rec {
          version = "8.2.3";
          src = super.fetchPypi {
            pname = "sphinx";
            inherit version;
            hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
          };
        });
      };
    };

    pythonEnv = python.withPackages (ps: with ps; [
      numpy matplotlib scikit-learn
      jupyterlab ipykernel
    ]);
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
      packages = [ pythonEnv ];
    };
  };
}
