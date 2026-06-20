from __future__ import annotations

from pathlib import Path

import openmc


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "bundle_smoke"
RUN_DIR.mkdir(parents=True, exist_ok=True)

FUEL_RADIUS_CM = 0.4096
PITCH_CM = 1.26
PIN_COUNT = 3
HEIGHT_CM = 2.0
FUEL_VOLUME_CM3 = PIN_COUNT * PIN_COUNT * 3.141592653589793 * FUEL_RADIUS_CM**2 * HEIGHT_CM


def build_bundle_model(
    fuel_temperature: float = 900.0,
    enrichment: float = 4.5,
    moderator_density: float = 0.997,
    moderator_temperature: float = 293.6,
    batches: int = 20,
    inactive: int = 5,
    particles: int = 2000,
    random_seed: int = 23,
) -> openmc.Model:
    fuel = openmc.Material(name=f"UO2 bundle fuel, {enrichment:.2f} wt% U-235")
    fuel.set_density("g/cm3", 10.4)
    fuel.add_element("U", 1.0, enrichment=enrichment)
    fuel.add_element("O", 2.0)
    fuel.temperature = fuel_temperature

    water = openmc.Material(name="Light water moderator")
    water.set_density("g/cm3", moderator_density)
    water.add_element("H", 2.0)
    water.add_element("O", 1.0)
    water.add_s_alpha_beta("c_H_in_H2O")
    water.temperature = moderator_temperature

    materials = openmc.Materials([fuel, water])

    half_width = PIN_COUNT * PITCH_CM / 2.0
    min_x = openmc.XPlane(x0=-half_width, boundary_type="reflective")
    max_x = openmc.XPlane(x0=half_width, boundary_type="reflective")
    min_y = openmc.YPlane(y0=-half_width, boundary_type="reflective")
    max_y = openmc.YPlane(y0=half_width, boundary_type="reflective")
    min_z = openmc.ZPlane(z0=-HEIGHT_CM / 2.0, boundary_type="reflective")
    max_z = openmc.ZPlane(z0=HEIGHT_CM / 2.0, boundary_type="reflective")
    box = +min_x & -max_x & +min_y & -max_y & +min_z & -max_z

    offsets = [-(PIN_COUNT - 1) * PITCH_CM / 2.0, 0.0, (PIN_COUNT - 1) * PITCH_CM / 2.0]
    fuel_cells: list[openmc.Cell] = []
    fuel_regions = []
    for ix, x0 in enumerate(offsets):
        for iy, y0 in enumerate(offsets):
            cyl = openmc.ZCylinder(x0=x0, y0=y0, r=FUEL_RADIUS_CM)
            fuel_region = -cyl & box
            fuel_regions.append(fuel_region)
            cell = openmc.Cell(name=f"fuel_pin_{ix}_{iy}", fill=fuel, region=fuel_region)
            cell.volume = 3.141592653589793 * FUEL_RADIUS_CM**2 * HEIGHT_CM
            fuel_cells.append(cell)

    moderator_region = box
    for region in fuel_regions:
        moderator_region &= ~region
    moderator_cell = openmc.Cell(name="bundle_moderator", fill=water, region=moderator_region)
    moderator_cell.volume = (PIN_COUNT * PITCH_CM) ** 2 * HEIGHT_CM - FUEL_VOLUME_CM3

    geometry = openmc.Geometry(fuel_cells + [moderator_cell])

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.batches = batches
    settings.inactive = inactive
    settings.particles = particles
    settings.seed = random_seed
    settings.temperature = {"method": "interpolation"}
    source_space = openmc.stats.Box(
        [-half_width, -half_width, -HEIGHT_CM / 2.0],
        [half_width, half_width, HEIGHT_CM / 2.0],
        only_fissionable=True,
    )
    settings.source = openmc.IndependentSource(space=source_space)

    fuel_filter = openmc.CellFilter(fuel_cells)
    moderator_filter = openmc.CellFilter(moderator_cell)

    fuel_response = openmc.Tally(name="fuel_response")
    fuel_response.filters = [fuel_filter]
    fuel_response.scores = ["flux", "fission", "(n,gamma)", "kappa-fission"]

    moderator_response = openmc.Tally(name="moderator_response")
    moderator_response.filters = [moderator_filter]
    moderator_response.scores = ["flux", "(n,gamma)"]

    return openmc.Model(
        geometry=geometry,
        materials=materials,
        settings=settings,
        tallies=openmc.Tallies([fuel_response, moderator_response]),
    )


if __name__ == "__main__":
    model = build_bundle_model()
    model.export_to_xml(directory=RUN_DIR)
    print(f"Wrote OpenMC bundle XML files to {RUN_DIR}")
