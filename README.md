# DTU Helsinki Asteroid Challenge (HAC)

This repostiory contains submission(s) to the Helsinki Asteroid Challenge by the team from the Technical University of Denmark (DTU).

The data is too large to comfortably fit within the Git repository.

## Dataset Setup

The dataset root is configured through the `DSDIR` environment variable. This variable is required by the Python code and by dataset scripts.

Set `DSDIR` to the directory where model folders (for example `AsteroidModel01_shape_public`) should live:

```sh
export DSDIR="/path/to/challenge_data"
```

After setting `DSDIR`, download the dataset with:

```sh
./bin/get_challenge_data.sh
```

The script can be run from any directory and will download and extract data into `DSDIR`.

## Contributor Information

- Add yourself as an author in `pyproject.toml`.
- Create a folder for your own experiments in `workspaces/YOUR_NAME`
- Contribute relevant bits to the common library in `src/hac`