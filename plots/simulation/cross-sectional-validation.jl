using Serialization
using CairoMakie
braak_labels = "Braak " .* ["1", "2/3", "4", "5", "6"]

begin
    f = Figure(size=(1000, 750))
    cols = Makie.wong_colors()
    for (j, tracer) in enumerate(["fbb", "fbp", "bf"])
        for i in 1:5
            data = deserialize(
                joinpath(OUTPUT_ROOT, tracer, "braak$(i)_trajectories.jls")
            )

            # Individual predictive trajectories
            data.subject_tau_sols
            data.subject_ab_sols

            # Posterior mean-parameter trajectories
            data.meanparam_tau_sols
            data.meanparam_ab_sols

            # Representative trajectory
            data.reference_tau_sol
            data.reference_ab_sol

            # Observations
            data.observations
            ax1 = Axis(f[j, i], xlabel="Tau SUVR", ylabel="Global Aβ SUVR")
            hidespines!(ax1, :r, :t)
            ax1.xticks=1.0:1.0:4.0

            if j == 1
                Label(f[0, i], braak_labels[i], tellwidth=false, font=:bold)
            end
            if j ==1
                xlims!(1.0, 3.5)
                ylims!(0.75, 1.2)
            elseif j == 2
                xlims!(1.0, 3.5)
                ylims!(0.7, 1.1)
            else
                xlims!(1.0, 3.5)
                ylims!(1.1, 1.8)
                ax1.yticks=1.1:0.2:1.7
            end
            if j < 3 
                hidexdecorations!(grid=false, ticks=false, )
            end
            
            
            if i != 1
                hideydecorations!(grid=false, ticks=false,)
            end
            
            for i in 1:1:1000 
                lines!(ax1, data.subject_tau_sols[:,i], data.subject_ab_sols[:,i],
                         color=(:grey, 0.025))
            end
            for i in 1:1:1000 
                lines!(ax1, data.meanparam_tau_sols[:,i], data.meanparam_ab_sols[:,i], 
                            color=(cols[1], 0.025))
            end
            scatter!(ax1, data.observations.tau_suvr, data.observations.amyloid_suvr, 
                    markersize=7.5, color=(cols[3], 0.7))
            lines!(ax1, data.reference_tau_sol, data.reference_ab_sol, 
                    color=(cols[2], 1.0), linewidth=1)
        end
    end
    # -------------------------------------------------------
    # Shared legend beneath all 15 panels
    # -------------------------------------------------------

    legend_elements = [
        LineElement(
            color=(:grey, 0.5),
            linewidth=5
        ),
        LineElement(
            color=(cols[1], 0.8),
            linewidth=5
        ),
        LineElement(
            color=cols[2],
            linewidth=5
        ),
        MarkerElement(
            marker=:circle,
            color=(cols[3], 0.8),
            markersize=15
        )
    ]

    legend_labels = [
        "Individual predictive trajectories",
        "Population mean parameter uncertainty",
        "Population mean parameter trajectory",
        "Observed PET"
    ]

    Legend(
        f[4, 1:5],
        legend_elements,
        legend_labels;
        orientation=:horizontal,
        framevisible=false,
        tellwidth=false,
        nbanks=2,
        labelsize=18,
        colgap=18
    )

    rowgap!(f.layout, 8)

    display(f)
end