using CairoMakie
using DrWatson
using Serialization
using Turing
using Colors
using ColorSchemes
using Distributions

pst = chainscat([deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-random-beta-lognormal-4x1000-$i.jls")) for i in 1:4]...)

begin
    using CairoMakie; CairoMakie.activate!()
    pst = chainscat([deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-random-beta-lognormal-4x1000-$i.jls")) for i in 1:4]...)
    bf_pst = chainscat([deserialize(projectdir("output/chains/population-atn/pst-samples-bf-random-lognormal-dense-1x1000-$i.jls")) for i in 1:4]...)
    
    colors = Makie.wong_colors();

    f = Figure(size=(1000, 1000), fontsize=20, figure_padding = 25)
    g = [f[i, 1] = GridLayout() for i in 1:6]

    # Ab production
    ax1 = Axis(g[1][1,1], xticks=0:0.1:0.5,   xlabel=L"1 / yr", xlabelsize=25,ylabel=L"\alpha", ylabelrotation=2pi, ylabelsize=30, ylabelpadding=20)
    hidespines!(ax1, :l, :t, :r)
    hideydecorations!(ax1, label=false)
    CairoMakie.xlims!(ax1, 0.0,0.5)
    hist!(vec(Array(pst[:Am_a])), bins=15, color=alphacolor(colors[1], 0.75), strokecolor=:white, strokewidth=1, label="ADNI")
    hist!(vec(Array(bf_pst[:Am_a])), bins=15, color=alphacolor(colors[3], 0.75), strokecolor=:white, strokewidth=1, label="BF")

    ax = Axis(g[1][1,2], xticks=0:0.4:1.2, xlabel=L"1 / yr", xlabelsize=25,)
    xlims!(ax, 0.0,1.44)
    ylims!(ax, 0.0,50)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:34
        density!(vec(Array(pst[Symbol("α_a[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    
    ax = Axis(g[1][1,3], xticks=0:0.4:1.2, xlabel=L"1 / yr", xlabelsize=25,)
    xlims!(ax, 0.0,1.44)
    ylims!(ax, 0.0,50)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:48
        density!(vec(Array(bf_pst[Symbol("α_a[$j]")])), color=alphacolor(colors[3], 0.5), strokecolor=:white, strokewidth=1)
    end

    # transport
    ax = Axis(g[2][1,1], xticks=0:0.03:0.15,  xlabel=L"1 / yr", xlabelsize=25,ylabel=L"\rho", ylabelrotation=2pi, ylabelsize=30, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.15)
    hist!(vec(Array(pst[:Pm_t])), bins=15, color=alphacolor(colors[1], 0.75), strokecolor=:white, strokewidth=1)
    hist!(vec(Array(bf_pst[:Pm_t])), bins=15, color=alphacolor(colors[3], 0.75), strokecolor=:white, strokewidth=1)

    ax = Axis(g[2][1,2], xticks=0:0.1:0.3, xlabel=L"1 / yr", xlabelsize=25,)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:34
        density!(vec(Array(pst[Symbol("ρ_t[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    xlims!(ax, 0.0,0.36)

    ax = Axis(g[2][1,3], xticks=0:0.1:0.3, xlabel=L"1 / yr", xlabelsize=25,)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:48
        density!(vec(Array(bf_pst[Symbol("ρ_t[$j]")])), color=alphacolor(colors[3], 0.5), strokecolor=:white, strokewidth=1)
    end
    xlims!(ax, 0.0,0.36)
    
    # tau production
    ax = Axis(g[3][1,1],  xticks=0:0.03:0.15,  xlabel=L"1 / yr", xlabelsize=25,ylabel=L"\gamma", ylabelrotation=2pi, ylabelsize=30, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.15)
    hist!(vec(Array(pst[:Am_t])), bins=15,color=alphacolor(colors[1], 0.75), strokecolor=:white, strokewidth=1)
    hist!(vec(Array(bf_pst[:Am_t])), bins=15,color=alphacolor(colors[3], 0.75), strokecolor=:white, strokewidth=1)
    
    ax = Axis(g[3][1,2], xticks=0:0.1:0.3,  xlabel=L"1 / yr", xlabelsize=25,)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:34
        density!(vec(Array(pst[Symbol("α_t[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    ylims!(ax, 0.0,250)
    xlims!(ax, 0.0,0.36)
    
    ax = Axis(g[3][1,3], xticks=0:0.1:0.3,  xlabel=L"1 / yr", xlabelsize=25,)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:48
        density!(vec(Array(bf_pst[Symbol("α_t[$j]")])), color=alphacolor(colors[3], 0.5), strokecolor=:white, strokewidth=1)
    end
    ylims!(ax, 0.0,250)
    xlims!(ax, 0.0,0.36)

    
    ax = Axis(g[4][1,1], xticks=0.0:0.05:0.25,  xlabel=L"1 / yr", xlabelsize=25,ylabel=L"\eta", ylabelrotation=2pi, ylabelsize=30, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.25)
    hist!(vec(Array(pst[:Em])), bins=15, color=alphacolor(colors[1], 0.75), strokecolor=:white, strokewidth=1)
    hist!(vec(Array(bf_pst[:Em])), bins=15, color=alphacolor(colors[3], 0.75), strokecolor=:white, strokewidth=1)
    
    ax = Axis(g[4][1,2], xticks=0:0.2:0.6, xlabel=L"1 / yr", xlabelsize=25,)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:34
        density!(vec(Array(pst[Symbol("η[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    xlims!(ax, 0.0,0.72)
    ax = Axis(g[4][1,3], xticks=0:0.2:0.6, xlabel=L"1 / yr", xlabelsize=25,)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:48
        density!(vec(Array(bf_pst[Symbol("η[$j]")])), color=alphacolor(colors[3], 0.5), strokecolor=:white, strokewidth=1)
    end
    xlims!(ax, 0.0,0.72)
    # coupling
    ax = Axis(g[5][1,1], xticks=4.5:0.5:6.5, xlabel=L"\beta \text{ FBB}", ylabelrotation=2pi, xlabelsize=30, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    hist!(vec(Array(pst[Symbol("β_fbb")])), bins=15, color=alphacolor(colors[1], 1.0), strokecolor=:white, strokewidth=1)
    xlims!(ax, 4.,7.)
    ax = Axis(g[5][1,2], xticks=4.5:0.5:6.5, xlabel=L"\beta \text{ FBP}", ylabelrotation=2pi, xlabelsize=30, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    hist!(vec(Array(pst[Symbol("β_fbp")])), bins=15, color=alphacolor(colors[1], 1.0), strokecolor=:white, strokewidth=1)
    xlims!(ax, 4.,7.)

    ax = Axis(g[5][1,3], xticks=4.5:0.5:6.5, xlabel=L"\beta \text{ FMM}", ylabelrotation=2pi, xlabelsize=30, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    hist!(vec(Array(bf_pst[Symbol("β")])), bins=10, color=alphacolor(colors[3], 1.0), strokecolor=:white, strokewidth=1)
    xlims!(ax, 4.,7.)

    colgap!.(g, 50)
                        # Legend(g2[3,:], elems, labels, framevisible = false, unique=true, labelsize=35, nbanks=5, tellheight=true, tellwidth=false)

    Legend(g[6][1,1:3], ax1, framevisible = false, labelsize=30, nbanks=2, tellheight=true, tellwidth=false)


    display(f)
end
save(projectdir("output/plots/inference/pst-adni-bf.png"),f)


begin
    using CairoMakie; CairoMakie.activate!()
    pst = chainscat([deserialize(projectdir("output/chains/population-atn/pst-samples-bf-random-lognormal-dense-1x1000-$i.jls")) for i in 1:4]...)
    
    colors = Makie.wong_colors();

    f = Figure(size=(1200, 1500), fontsize=20, figure_padding = 25)
    g = [f[i, 1] = GridLayout() for i in 1:5]

    ax = Axis(g[1][1,1], xticks=0:0.1:0.5,   xlabel=L"1 / yr", xlabelsize=25,ylabel=L"\alpha", ylabelrotation=2pi, ylabelsize=30, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    CairoMakie.xlims!(ax, 0.0,0.5)
    hist!(vec(Array(pst[:Am_a])), bins=15, color=alphacolor(colors[1], 1.0), strokecolor=:white, strokewidth=1)

    ax = Axis(g[1][1,2:3], xticks=0:0.25:1.25, xlabel=L"1 / yr", xlabelsize=25,)
    xlims!(ax, 0.0,1.5)
    ylims!(ax, 0.0,50)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:48
        density!(vec(Array(pst[Symbol("α_a[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    
    ax = Axis(g[2][1,1], xticks=0:0.03:0.15,  xlabel=L"1 / yr", xlabelsize=25,ylabel=L"\rho", ylabelrotation=2pi, ylabelsize=30, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.15)
    hist!(vec(Array(pst[:Pm_t])), bins=15, color=alphacolor(colors[1], 1.0), strokecolor=:white, strokewidth=1)

    ax = Axis(g[2][1,2:3], xticks=0:0.05:0.30, xlabel=L"1 / yr", xlabelsize=25,)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:48
        density!(vec(Array(pst[Symbol("ρ_t[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    xlims!(ax, 0.0,0.36)
    
    ax = Axis(g[3][1,1],  xticks=0:0.03:0.15,  xlabel=L"1 / yr", xlabelsize=25,ylabel=L"\gamma", ylabelrotation=2pi, ylabelsize=30, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.15)
    hist!(vec(Array(pst[:Am_t])), bins=15,color=alphacolor(colors[1], 1.0), strokecolor=:white, strokewidth=1)
    
    ax = Axis(g[3][1,2:3], xticks=0:0.05:0.30,  xlabel=L"1 / yr", xlabelsize=25,)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:48
        density!(vec(Array(pst[Symbol("α_t[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    ylims!(ax, 0.0,250)
    xlims!(ax, 0.0,0.36)
    
    ax = Axis(g[5][1,1], xticks=4.5:0.5:6.5, xlabel=L"\beta \text{ FMM}", ylabelrotation=2pi, xlabelsize=30, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    hist!(vec(Array(pst[Symbol("β")])), bins=15, color=alphacolor(colors[1], 1.0), strokecolor=:white, strokewidth=1)
    xlims!(ax, 4.,7.)

    ax = Axis(g[4][1,1], xticks=0.0:0.05:0.25,  xlabel=L"1 / yr", xlabelsize=25,ylabel=L"\eta", ylabelrotation=2pi, ylabelsize=30, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.25)
    hist!(vec(Array(pst[:Em])), bins=15, color=alphacolor(colors[1], 1.0), strokecolor=:white, strokewidth=1)
    ax = Axis(g[4][1,2:3], xticks=0:0.1:0.6, xlabel=L"1 / yr", xlabelsize=25,)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:48
        density!(vec(Array(pst[Symbol("η[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    xlims!(ax, 0.0,0.72)

    colgap!.(g, 50)

    display(f)
end
