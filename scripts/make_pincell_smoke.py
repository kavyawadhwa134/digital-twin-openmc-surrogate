from pathlib import Path

import openmc


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "pincell_smoke"
RUN_DIR.mkdir(parents=True, exist_ok=True)


def build_pincell_model(
    fuel_temperature: float = 900.0,
    enrichment: float = 4.95,
    moderator_density: float = 0.997,
    moderator_temperature: float = 293.6,
    fuel_radius: float = 0.4096,
    pin_pitch: float = 1.26,
    cladding_thickness: float = 0.057,
    boron_ppm: float = 0.0,
    batches: int = 20,
    inactive: int = 5,
    particles: int = 2000,
    random_seed: int = 7,
) -> openmc.Model:
    if fuel_radius <= 0.0:
        raise ValueError("fuel_radius must be positive")
    if pin_pitch <= 0.0:
        raise ValueError("pin_pitch must be positive")
    if cladding_thickness < 0.0:
        raise ValueError("cladding_thickness cannot be negative")
    clad_outer_radius = fuel_radius + cladding_thickness
    if 2.0 * clad_outer_radius >= pin_pitch:
        raise ValueError(
            "pin_pitch must be larger than twice the clad outer radius "
            f"({pin_pitch=} {fuel_radius=} {cladding_thickness=})"
        )

    fuel = openmc.Material(name=f"UO2 fuel, {enrichment:.2f} wt% U-235")
    fuel.set_density("g/cm3", 10.4)
    fuel.add_element("U", 1.0, enrichment=enrichment)
    fuel.add_element("O", 2.0)
    fuel.temperature = fuel_temperature

    water = openmc.Material(name="Light water moderator")
    water.set_density("g/cm3", moderator_density)
    water.add_element("H", 2.0)
    water.add_element("O", 1.0)
    if boron_ppm > 0.0:
        # Approximate soluble natural boron as a trace atomic fraction.
        water.add_element("B", boron_ppm * 1.0e-6)
    water.add_s_alpha_beta("c_H_in_H2O")
    water.temperature = moderator_temperature

    cladding = openmc.Material(name="Zircaloy-like cladding")
    cladding.set_density("g/cm3", 6.55)
    cladding.add_element("Zr", 1.0)
    cladding.temperature = moderator_temperature

    materials = openmc.Materials([fuel, water, cladding])

    fuel_surface = openmc.ZCylinder(r=fuel_radius)
    clad_outer_surface = openmc.ZCylinder(r=clad_outer_radius)
    half_pitch = pin_pitch / 2.0
    min_x = openmc.XPlane(x0=-half_pitch, boundary_type="reflective")
    max_x = openmc.XPlane(x0=half_pitch, boundary_type="reflective")
    min_y = openmc.YPlane(y0=-half_pitch, boundary_type="reflective")
    max_y = openmc.YPlane(y0=half_pitch, boundary_type="reflective")
    min_z = openmc.ZPlane(z0=-1.0, boundary_type="reflective")
    max_z = openmc.ZPlane(z0=1.0, boundary_type="reflective")

    box = +min_x & -max_x & +min_y & -max_y & +min_z & -max_z
    fuel_cell = openmc.Cell(name="fuel", fill=fuel, region=-fuel_surface & box)
    cladding_cell = openmc.Cell(
        name="cladding", fill=cladding, region=+fuel_surface & -clad_outer_surface & box
    )
    moderator_cell = openmc.Cell(
        name="moderator", fill=water, region=+clad_outer_surface & box
    )
    fuel_cell.volume = 3.141592653589793 * fuel_radius**2 * 2.0
    cladding_cell.volume = 3.141592653589793 * (
        clad_outer_radius**2 - fuel_radius**2
    ) * 2.0
    moderator_cell.volume = (pin_pitch * pin_pitch * 2.0) - fuel_cell.volume - cladding_cell.volume
    geometry = openmc.Geometry([fuel_cell, cladding_cell, moderator_cell])

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.batches = batches
    settings.inactive = inactive
    settings.particles = particles
    settings.seed = random_seed
    settings.temperature = {"method": "interpolation"}
    source_space = openmc.stats.Box(
        [-fuel_radius, -fuel_radius, -1.0],
        [fuel_radius, fuel_radius, 1.0],
    )
    settings.source = openmc.IndependentSource(
        space=source_space,
        constraints={"fissionable": True},
    )

    fuel_filter = openmc.CellFilter(fuel_cell)
    moderator_filter = openmc.CellFilter(moderator_cell)

    fuel_response = openmc.Tally(name="fuel_response")
    fuel_response.filters = [fuel_filter]
    fuel_response.scores = ["flux", "fission", "(n,gamma)", "kappa-fission"]

    moderator_response = openmc.Tally(name="moderator_response")
    moderator_response.filters = [moderator_filter]
    moderator_response.scores = ["flux", "(n,gamma)"]

    tallies = openmc.Tallies([fuel_response, moderator_response])

    return openmc.Model(
        geometry=geometry,
        materials=materials,
        settings=settings,
        tallies=tallies,
    )


def build_model() -> openmc.Model:
    return build_pincell_model()


if __name__ == "__main__":
    model = build_model()
    model.export_to_xml(directory=RUN_DIR)
    print(f"Wrote OpenMC input XML files to {RUN_DIR}")
