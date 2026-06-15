{
  description = "Python ML dev environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs   = nixpkgs.legacyPackages.${system};

    # python312 supports sphinx 9.x — no conflict
    pythonEnv = pkgs.python312.withPackages (ps: with ps; [
      numpy
      matplotlib
      scikit-learn
      jupyterlab
      ipykernel
    ]);
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = [ pythonEnv ];
      shellHook = ''
        echo "Python: $(python --version)"
        echo "Jupyter: $(jupyter --version | head -1)"
      '';
    };
  };
}
