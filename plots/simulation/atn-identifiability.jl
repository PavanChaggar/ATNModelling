using DrWatson, CSV, DataFrames, DelimitedFiles, Serialization
using Turing
using CairoMakie, ColorSchemes, Colors
CairoMakie.activate!()
begin
    ts_slide = [collect(range(0 +i, 3 + i, 3)) for i in 0:3:27]
    psts = Vector{Chains}()
    for t in ts_slide
        _ts = Int.(extrema(t))
        _pst = deserialize(projectdir("output/chains/atn-identifiability/pst-t$(_ts[1])-$(_ts[1])-$(_ts[end]).jls"))
        println((sum(_pst[:numerical_error])))
        push!(psts, _pst)
    end
end

cmap = colormap("Blues", 22);
begin
    f = Figure(size=(1000, 1500), fontsize=25)
    ax = Axis(f[1, 1], xticks=(0:3:30),
    xlabel=L"t_0", xlabelsize=35, ylabel=L"\alpha", ylabelrotation=2pi, ylabelsize=35,
    yticklabelsize=20, yticks=0.5:0.1:1.0, ylabelpadding=30)
    hideydecorations!(ax, label=false, grid=false, ticklabels=false)
    hidexdecorations!(ax, grid=false, ticks=false,)
    hidespines!(ax, :t, :r, :l)
    CairoMakie.ylims!(ax, 0.5,1.02)
    for (i, pst) in enumerate(psts)
        CairoMakie.hist!(vec(Array(pst[:α_a])), bins=50, offset=3i -3 , color=(cmap[end-3], 0.75), scale_to=1.75, direction=:x)
    end
    
    ax = Axis(f[2, 1], xticks=(0:3:30),
    xlabel=L"t_0", xlabelsize=35, ylabel=L"\rho", ylabelrotation=2pi, ylabelsize=35,
    yticklabelsize=20, ylabelpadding=25, yticks=0:0.02:0.1)
    hideydecorations!(ax, label=false, grid=false, ticklabels=false)
    hidexdecorations!(ax, grid=false, ticks=false,)
    hidespines!(ax, :t, :r, :l)
    CairoMakie.ylims!(ax, 0.0,0.11)
    for (i, pst) in enumerate(psts)
        CairoMakie.hist!(vec(Array(pst[:ρ_t])), bins=50, offset=3i -3 , color=(cmap[end-3], 0.75), scale_to=1.75, direction=:x)
    end

    ax = Axis(f[3, 1], xticks=(0:3:30),
    xlabel=L"t_0", xlabelsize=35, ylabel=L"\gamma", ylabelrotation=2pi, ylabelsize=35,
    yticklabelsize=20, yticks=0:0.2:1., ylabelpadding=35)
    hideydecorations!(ax, label=false, grid=false, ticklabels=false)
    hidexdecorations!(ax, grid=false, ticks=false,)
    hidespines!(ax, :t, :r, :l)
    CairoMakie.ylims!(ax, 0.0,1.1)
    for (i, pst) in enumerate(psts)
        CairoMakie.hist!(vec(Array(pst[:α_t])), bins=50, offset=3i -3 , color=(cmap[end-3], 0.75), scale_to=1.75, direction=:x)
    end

    ax = Axis(f[4, 1], xticks=(0:3:30),
    xlabel=L"t_0", xlabelsize=35, ylabel=L"\beta", ylabelrotation=2pi, ylabelsize=35,
    yticklabelsize=20, yticks=2:0.5:5, ylabelpadding=45)
    hideydecorations!(ax, label=false, grid=false, ticklabels=false)
    hidexdecorations!(ax, grid=false, ticks=false,)
    hidespines!(ax, :t, :r, :l)
    CairoMakie.ylims!(ax, 2.0, 5.1)
    for (i, pst) in enumerate(psts)
        CairoMakie.hist!(vec(Array(pst[:β])), bins=50, offset=3i -3 , color=(cmap[end-3], 0.75), scale_to=1.75, direction=:x)
    end

    ax = Axis(f[5, 1], xticks=(0:3:30),
    xlabel="Time / years", xlabelsize=30, ylabel=L"\eta", ylabelrotation=2pi, ylabelsize=35,
    xticklabelsize=20, yticklabelsize=20, yticks=0:0.02:0.15, ylabelpadding=20)
    hideydecorations!(ax, label=false, grid=false, ticklabels=false)
    hidespines!(ax, :t, :r, :l)
    CairoMakie.ylims!(ax, 0.0,0.151)
    for (i, pst) in enumerate(psts)
        CairoMakie.hist!(vec(Array(pst[:η])), bins=50, offset=3i -3 , color=(cmap[end-3], 0.75), scale_to=1.75, direction=:x)
    end
    # vlines!(ax, 0.01, color=(:red, 0.75), linewidth=2.0)

    display(f)
end
save(projectdir("output/plots/simulation/atn-identifiability.pdf"), f)


begin
    f = Figure(size=(800, 300), fontsize=15)
    ax = Axis(f[1, 1], xticks=(0:3:30),
    xlabel=L"t_0", xlabelsize=25, ylabel=L"\beta", ylabelrotation=2pi, ylabelsize=25, yticks=2:0.5:5, ylabelpadding=45)
    hideydecorations!(ax, label=false, grid=false, ticklabels=false)
    # hidexdecorations!(ax, grid=false, ticks=false,)
    hidespines!(ax, :t, :r, :l)
    CairoMakie.ylims!(ax, 2.0, 5.1)
    for (i, pst) in enumerate(psts)
        CairoMakie.hist!(vec(Array(pst[:β])), bins=50, offset=3i -3 , color=(cmap[end-3], 0.75), scale_to=1.75, direction=:x)
    end
    display(f)
end
save(projectdir("output/plots/simulation/atn-identifiability-beta.pdf"), f)
