using CairoMakie
using DrWatson
using Serialization
using Turing
using Colors
using ColorSchemes
using Distributions

pst = deserialize(projectdir("output/chains/population-scaled-atn/pst-samples-harmonised-dense-1x1000.jls"));

begin
    using CairoMakie; CairoMakie.activate!()
    pst = deserialize(projectdir("output/chains/population-scaled-atn/pst-samples-harmonised-dense-1x1000.jls"));
    
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
    for j in 1:44
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
    for j in 1:44
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
    for j in 1:44
        density!(vec(Array(pst[Symbol("α_t[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    ylims!(ax, 0.0,250)
    xlims!(ax, 0.0,0.315)
    
    ax = Axis(g[5][1,1], xticks=4.5:0.5:6., ylabel=L"\beta", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    for j in 1:44
        hist!(vec(Array(pst[Symbol("β")])), bins=15, color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    xlims!(ax, 3,6.)
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
    for j in 1:44
        density!(vec(Array(pst[Symbol("η[$j]")])), color=alphacolor(colors[1], 0.5), strokecolor=:white, strokewidth=1)
    end
    xlims!(ax, 0.0,0.84)

    colgap!.(g, 50)

    display(f)
end

ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);
abcmap = ColorScheme(ab_c);
taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
atrcmap = ColorScheme(atr_c);
begin

    using CairoMakie; CairoMakie.activate!()
    pst = deserialize(projectdir("output/chains/population-scaled-atn/pst-samples-harmonised-dense-1x1000.jls"));
    
    colors = Makie.wong_colors();
    titlesize=20
    f = Figure(size=(1200, 300), fontsize=20, figure_padding = 25, )
    g = [f[1, i] = GridLayout() for i in 1:5]

    ax = Axis(g[1][1,1], xticks=0:0.2:0.4,   xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, title=L"\alpha \text{: A\beta} \text{ production}", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    CairoMakie.xlims!(ax, 0.0,0.5)
    hist!(vec(Array(pst[:Am_a])), bins=15, color=alphacolor(get(abcmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    
    ax = Axis(g[2][1,1], xticks=0:0.04:0.08,  xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, title=L"\rho \text{: tau transport}", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.1)
    hist!(vec(Array(pst[:Pm_t])), bins=15, color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    
    ax = Axis(g[3][1,1],  xticks=0:0.05:0.1,  xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, title=L"\gamma \text{: tau production}", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.125)
    hist!(vec(Array(pst[:Am_t])), bins=15,color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    

    ax = Axis(g[4][1,1], xticks=3:1:6.5, titlesize=titlesize, title=L"\beta \text{: A}\beta/\text{tau coupling}", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 3,6.5)
    hist!(vec(Array(pst[:β])), bins=15,color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)

    ax = Axis(g[5][1,1], xticks=0.0:0.05:0.1,  xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, title=L"\eta \text{: atrophy rate}", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.125)
    hist!(vec(Array(pst[:Em])), bins=15, color=alphacolor(get(atrcmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    colgap!(f.layout, 20)

    display(f)
end
save(projectdir("output/plots/inference-results/pst-harmonised-scaled.pdf"),f)


ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);
abcmap = ColorScheme(ab_c);
taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
atrcmap = ColorScheme(atr_c);
begin

    using CairoMakie; CairoMakie.activate!()
    pst = deserialize(projectdir("output/chains/population-scaled-atn/pst-samples-harmonised-dense-1x1000.jls"));
    bf_pst = deserialize(projectdir("output/bf-output/bf/pst-samples-scaled-fixed-1x1000.jls"));

    colors = Makie.wong_colors();
    titlesize=25
    f = Figure(size=(1200, 400), fontsize=20, figure_padding = 25, )
    g = [f[1, i] = GridLayout() for i in 1:5]
    g2 = [f[2, i] = GridLayout() for i in 1:5]

    ax = Axis(g[1][1,1], xticks=0:0.2:0.4,   xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, title=L"\alpha \text{: A\beta} \text{ production}", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    CairoMakie.xlims!(ax, 0.0,0.5)
    hist!(vec(Array(pst[:Am_a])), bins=15, color=alphacolor(get(abcmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    hidexdecorations!(ax, grid=false, ticks=false)

    ax = Axis(g[2][1,1], xticks=0:0.04:0.08,  xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, title=L"\rho \text{: tau transport}", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.1)
    hist!(vec(Array(pst[:Pm_t])), bins=15, color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    hidexdecorations!(ax, grid=false, ticks=false)

    ax = Axis(g[3][1,1],  xticks=0:0.05:0.1,  xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, title=L"\gamma \text{: tau production}", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.225)
    hist!(vec(Array(pst[:Am_t])), bins=15,color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    hidexdecorations!(ax, grid=false, ticks=false)


    ax = Axis(g[4][1,1], xticks=3:1:8.5, titlesize=titlesize, title=L"\beta \text{: A}\beta/\text{tau coupling}", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 3,8.5)
    hist!(vec(Array(pst[:β])), bins=15,color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    hidexdecorations!(ax, grid=false, ticks=false)

    ax = Axis(g[5][1,1], xticks=0.0:0.05:0.1,  xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, title=L"\eta \text{: atrophy rate}", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    hidexdecorations!(ax, grid=false, ticks=false)
    xlims!(ax, 0.0,0.125)
    hist!(vec(Array(pst[:Em])), bins=15, color=alphacolor(get(atrcmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    colgap!(f.layout, 20)



    ax = Axis(g2[1][1,1], xticks=0:0.2:0.4,   xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    CairoMakie.xlims!(ax, 0.0,0.5)
    hist!(vec(Array(bf_pst[:Am_a])), bins=15, color=alphacolor(get(abcmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    
    ax = Axis(g2[2][1,1], xticks=0:0.04:0.08,  xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.1)
    hist!(vec(Array(bf_pst[:Pm_t])), bins=15, color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    
    ax = Axis(g2[3][1,1],  xticks=0:0.05:0.1,  xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.225)
    hist!(vec(Array(bf_pst[:Am_t])), bins=15,color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    

    # ax = Axis(g2[4][1,1], xticks=3:1:8.5, titlesize=titlesize, ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    # hidespines!(ax, :l, :t, :r)
    # hideydecorations!(ax, label=false)
    # xlims!(ax, 3,8.5)
    # hist!(vec(Array(bf_pst[:β])), bins=15,color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)

    ax = Axis(g2[5][1,1], xticks=0.0:0.05:0.1,  xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
    hidespines!(ax, :l, :t, :r)
    hideydecorations!(ax, label=false)
    xlims!(ax, 0.0,0.125)
    hist!(vec(Array(bf_pst[:Em])), bins=15, color=alphacolor(get(atrcmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    colgap!(f.layout, 20)


    display(f)
end
save(projectdir("output/plots/inference-results/pst-harmonised-scaled-adni-bf.pdf"),f)
