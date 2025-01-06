using CairoMakie
using DrWatson
using Serialization
using Turing
using Colors
using Distributions


begin
    using CairoMakie; CairoMakie.activate!()
    pst = deserialize(projectdir("output/chains/population-atn/pst-samples-lognormal-2-1x1000.jls"));
    
    colors = Makie.wong_colors();

    f = Figure(size=(1200, 1500), fontsize=20, figure_padding = 25)
    g = [f[i, 1] = GridLayout() for i in 1:5]

    ax = Axis(g[1][1,1], xticks=0:0.1:0.35,   xlabel=L"1 / yr", xlabelsize=25,ylabel=L"\alpha", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    CairoMakie.xlims!(ax, 0.0,0.35)
    hist!(vec(Array(pst[:Am_a])), bins=15, color=alphacolor(colors[1], 1.0), strokecolor=:white, strokewidth=1)

    ax = Axis(g[1][1,2:3], xticks=0:0.25:1.0, xlabel=L"1 / yr", xlabelsize=25,)
    xlims!(ax, 0.0,1.05)
    ylims!(ax, 0.0,150)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:22
        density!(vec(Array(pst[Symbol("α_a[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    
    ax = Axis(g[2][1,1], xticks=0:0.025:0.075,  xlabel=L"1 / yr", xlabelsize=25,ylabel=L"\rho", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.0875)
    hist!(vec(Array(pst[:Pm_t])), bins=15, color=alphacolor(colors[1], 1.0), strokecolor=:white, strokewidth=1)

    ax = Axis(g[2][1,2:3], xticks=0:0.05:0.35, xlabel=L"1 / yr", xlabelsize=25,)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:22
        density!(vec(Array(pst[Symbol("ρ_t[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    xlims!(ax, 0.0,0.315)
    
    ax = Axis(g[3][1,1],  xticks=0:0.1:0.35,  xlabel=L"1 / yr", xlabelsize=25,ylabel=L"\gamma", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.15)
    hist!(vec(Array(pst[:Am_t])), bins=15,color=alphacolor(colors[1], 1.0), strokecolor=:white, strokewidth=1)
    
    ax = Axis(g[3][1,2:3], xticks=0:0.05:0.35,  xlabel=L"1 / yr", xlabelsize=25,)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:22
        density!(vec(Array(pst[Symbol("α_t[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    ylims!(ax, 0.0,250)
    xlims!(ax, 0.0,0.315)
    
    ax = Axis(g[5][1,1], xticks=4.5:0.5:6., ylabel=L"\beta", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    for j in 1:37
        hist!(vec(Array(pst[Symbol("β")])), bins=15, color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    xlims!(ax, 4.5,6.25)
    ax = Axis(g[5][1,2:3])
    hidespines!(ax)
    hideydecorations!(ax)
    hidexdecorations!(ax)
    
    ax = Axis(g[4][1,1], xticks=0.0:0.05:0.15,  xlabel=L"1 / yr", xlabelsize=25,ylabel=L"\eta", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.175)
    hist!(vec(Array(pst[:Em])), bins=15, color=alphacolor(colors[1], 1.0), strokecolor=:white, strokewidth=1)
    ax = Axis(g[4][1,2:3], xticks=0:0.2:0.8, xlabel=L"1 / yr", xlabelsize=25,)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax)
    for j in 1:22
        density!(vec(Array(pst[Symbol("η[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    xlims!(ax, 0.0,0.84)

    colgap!.(g, 50)

    display(f)
end

pst = chainscat([deserialize("/Users/pavanchaggar/Projects/abtau/analysis/output/chains/beta-chains/031124/atn-noatr-beta-$i-1000-ln.jls") for i in 1:4]...)