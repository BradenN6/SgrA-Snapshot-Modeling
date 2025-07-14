module dc

# Model for low-luminosity active galactic nuclei (LLAGN)
# see https://dominic-chang.com/Krang.jl/v0.4.0/examples/polarization-example
# Shows the n = 0 (direct) and n = 1 (indirect) photons emitted from the
# source, viewed at a fixed inclination angle from the BH spin axis.

using Krang
using CairoMakie
using Printf

# Plotting theme
curr_theme = Theme(
    Axis = (
        xticksvisible = false,
        xticklabelsvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
    ),
)
set_theme!(merge!(curr_theme, theme_latexfonts()))

macro var2string(var)
    :($(esc(var)) = $(String(var)))
end


function dualcone(spin, θo, ρmax, χ, ι, βv, σ, η1, η2, R, p1, p2,
    θs, vary; plot_polarized=false)

    metric = Krang.Kerr(spin);

    vdict = Dict([("spin", spin), ("θo", θo), ("ρmax", ρmax),
        ("χ", χ), ("ι", ι), ("βv", βv), ("σ", σ), ("η1", η1),
        ("η2", η2), ("R", R), ("p1", p1), ("p2", p2), ("θs", θs)])

    # camera, resolution 400x400 pixels
    camera = Krang.IntensityCamera(metric, θo, -ρmax, ρmax, -ρmax, ρmax, 400);

    # Render scene with mesh objects of
    # ElectronSynchrotronPowerLawPolarization material
    # with parameters used to define magnetic field and fluid velocity
    magfield1 = Krang.SVector(sin(ι) * cos(η1), sin(ι) * sin(η1), cos(ι));
    magfield2 = Krang.SVector(sin(ι) * cos(η2), sin(ι) * sin(η2), cos(ι));
    vel = Krang.SVector(βv, (π / 2), χ);

    # mesh geometries
    material1 = Krang.ElectronSynchrotronPowerLawPolarization(
        magfield1...,
        vel...,
        σ,
        R,
        p1,
        p2,
        (0, 1),
    );
    geometry1 = Krang.ConeGeometry((θs))
    material2 = Krang.ElectronSynchrotronPowerLawPolarization(
        magfield2...,
        vel...,
        σ,
        R,
        p1,
        p2,
        (0, 1),
    );
    geometry2 = Krang.ConeGeometry((π - θs))

    # One mesh for each geometry
    # scene from both meshes
    mesh1 = Krang.Mesh(geometry1, material1)
    mesh2 = Krang.Mesh(geometry2, material2)

    # Render scene with camera and plot Stokes parameters
    scene = Krang.Scene((mesh1, mesh2))
    stokesvals = render(camera, scene)

    if plot_polarized
        fig = Figure(resolution = (700, 700));

        ax1 = Axis(fig[1, 1], aspect = 1, title = "I")
        ax2 = Axis(fig[1, 2], aspect = 1, title = "Q")
        ax3 = Axis(fig[2, 1], aspect = 1, title = "U")
        ax4 = Axis(fig[2, 2], aspect = 1, title = "V")
        colormaps = [:afmhot, :redsblues, :redsblues, :redsblues]

        # vectorized function applications

        zip(
            [ax1, ax2, ax3, ax4],
            [
                getproperty.(stokesvals, :I),
                getproperty.(stokesvals, :Q),
                getproperty.(stokesvals, :U),
                getproperty.(stokesvals, :V),
            ],
            colormaps,
        ) .|> x -> heatmap!(x[1], x[2], colormap = x[3])
    else
        fig = Figure(resolution = (700, 700));
        ax1 = Axis(fig[1, 1], aspect = 1, title = "I")
        colormaps = [:afmhot]

        heatmap!(ax1, getproperty.(stokesvals, :I), colormap = :afmhot)
    end

    if vary == "spin" || vary == "θo"
        title = @sprintf("Spin = %.2f, θ₀ = %05.2f", spin, θo * 180 / π)
        supertitle = Label(fig[0, :], title, 
            fontsize = 30, 
            tellwidth=false,
            halign=:center)
    else
        vary_var = get(vdict, vary, spin)
        title = @sprintf("Spin = %.2f, θ₀ = %05.2f, %s = %07.4f", spin, θo * 180 / π, vary, vary_var)
        supertitle = Label(fig[0, :],
            title, 
            fontsize = 30, 
            tellwidth=false,
            halign=:center)
    end

    # show figure
    fig

    if vary == "spin" || vary == "θo"
        fname = @sprintf("polarization_a=%.2f_θo=%05.2f.png", spin, θo * 180 / π)
        save(fname, fig)
    else
        vary_var = get(vdict, vary, spin)
        fname = @sprintf("polarization_a=%.2f_θo=%04.2f_%s=%07.4f.png", spin, θo * 180 / π, vary, vary_var)
        save(fname, fig)
    end

end

function main()
    # Defaults
    # Kerr BH, spin=0.94, asymptotic observer at inclination θo = 17deg
    spin=0.94
    θo = 17 * π / 180;
    ρmax = 10.0;

    # Fluid Parameters
    χ = -1.705612782769303
    ι = 0.5807355065517938
    βv = 0.8776461626924748
    σ = 0.7335172899224874
    η1 = 2.6444786738735804
    η2 = π - η1

    # Emissivity Profile
    R = 3.3266154761905455
    p1 = 4.05269835622511
    p2 = 4.411852974336667

    # Mesh Geometries
    # ConeGeometry, Opening Angle 75deg
    # (0, 1) - raytrace n=0 and n=1 subimages
    θs = (75 * π / 180)

    spin_range = [0.01:0.01:0.99;]
    neg_spin_range = [-0.01:-0.01:-0.99;]
    θo_range = [1 * π / 180:π / 128:89 * π / 180;]
    χ_range = [-π:0.1:0.0;]
    ι_range = [0.0:0.02:π/2;]
    βv_range = [0.0:0.01:0.9;]
    θs_range = [1 * π/180: π/128: 89 * π/180;]

    # vary spin
    #=
    for spin in spin_range
        dualcone(spin, θo, ρmax, χ, ι, βv, σ, η1, η2, R, p1, p2, θs, "spin")
    end
    
    for neg_spin in neg_spin_range
        dualcone(neg_spin, θo, ρmax, χ, ι, βv, σ, η1, η2, R, p1, p2, θs, "spin")
    end

    for θo in θo_range
        dualcone(spin, θo, ρmax, χ, ι, βv, σ, η1, η2, R, p1, p2, θs, "θo")
    end

    for χ in χ_range
        dualcone(spin, θo, ρmax, χ, ι, βv, σ, η1, η2, R, p1, p2, θs, "χ")
    end

    for ι in ι_range
        dualcone(spin, θo, ρmax, χ, ι, βv, σ, η1, η2, R, p1, p2, θs, "ι")
    end

    for βv in βv_range
        dualcone(spin, θo, ρmax, χ, ι, βv, σ, η1, η2, R, p1, p2, θs, "βv")
    end
    =#

    for θs in θs_range
        dualcone(spin, θo, ρmax, χ, ι, βv, σ, η1, η2, R, p1, p2, θs, "θs")
    end
    
end

end

dc.main()