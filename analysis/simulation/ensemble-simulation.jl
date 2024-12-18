
# function make_prob_func(initial_conditions, ρ_t, α_a, α_t, β, η, _times)
#     function prob_func(prob,i,repeat)
#         remake(prob, u0=initial_conditions[i], 
#                      p=[ρ_t[i], α_a[i], α_t[i], β, η[i]], saveat=_times[i])
#     end
# end
# function output_func(sol,i)
#     (sol,false)
# end

# function get_retcodes(es)
#     [SciMLBase.successful_retcode(sol) for sol in es]
# end

# function success_condition(retcodes)
#     allequal(retcodes) && retcodes[1] == 1
# end
# function split_sols(esol, ab_idx, tau_idx)
#     d = [[vec(s[1:72, a_idx]), vec(s[73:144, t_idx]), vec(s[145:216, t_idx])] 
#           for (s, a_idx, t_idx) in zip(esol, ab_idx, tau_idx)]
#     ab = reduce(vcat, [_d[1] for _d in d])
#     tau = reduce(vcat, [_d[2] for _d in d])
#     vol = reduce(vcat, [_d[3] for _d in d])     
#     return ab, tau, vol
# end
# ensemble_prob = EnsembleProblem(prob,
#                                     prob_func=make_prob_func(inits, 
#                                     fill(fill(1.0, n_subjects), 3)..., 5.0, ones(n_subjects), ts), 
#                                     output_func=output_func)
# esol = solve(ensemble_prob, Tsit5(), trajectories=n_subjects)

# @model function ensemble_fit(ab_data, tau_data, vol_data, prob, inits, times, ab_tidx, tau_tidx, n)
#     # @model function ensemble_fit(prob, inits, times, ab_tidx, tau_tidx, n)
    
#         σ_a  ~ InverseGamma(2,3)
#         σ_t  ~ InverseGamma(2,3)
#         σ_v  ~ InverseGamma(2,3)
        
#         Am_a ~ truncated(Normal(), lower=0)
#         As_a ~ truncated(Normal(), lower=0)
    
#         Pm_t ~ truncated(Normal(), lower=0)
#         Ps_t ~ truncated(Normal(), lower=0)
        
#         Am_t ~ truncated(Normal(), lower=0)
#         As_t ~ truncated(Normal(), lower=0)
    
#         Em   ~ truncated(Normal(), lower=0)
#         Es   ~ truncated(Normal(), lower=0)
        
#         β    ~ truncated(Normal(3, 1), lower=0)
    
#         α_a  ~ filldist(truncated(Normal(Am_a, As_a), lower=0), n)
#         ρ_t  ~ filldist(truncated(Normal(Pm_t, Ps_t), lower=0), n)
#         α_t  ~ filldist(truncated(Normal(Am_t, As_t), lower=0), n)
#         η    ~ filldist(truncated(Normal(Em, Es), lower=0), n)
    
#         for i in eachindex(1:n)
#             _prob = remake(prob, u0 = inits[i], p = [α_a[i], ρ_t[i], α_t[i], β, η[i]])
#             _sol = solve(_prob, Tsit5(), abstol = 1e-6, reltol = 1e-6, 
#                          sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), saveat=times[i])
#             if !successful_retcode(_sol)
#                 Turing.@addlogprob! -Inf
#                 println("failed")
#                 break
#             end
#             ab_preds, tau_preds, vol_preds = split_sols_2(_sol, ab_tidx[i], tau_tidx[i])
#             # ab_data[i] ~ MvNormal(ab_preds, σ_a^2 * I)
#             # tau_data[i] ~ MvNormal(tau_preds, σ_t^2 * I)
#             # vol_data[i] ~ MvNormal(vol_preds, σ_v^2 * I) 
#             Turing.@addlogprob! loglikelihood(MvNormal(ab_preds, σ_a^2 * I),  ab_data[i])
#             Turing.@addlogprob! loglikelihood(MvNormal(tau_preds, σ_t^2 * I),  tau_data[i])
#             Turing.@addlogprob! loglikelihood(MvNormal(vol_preds, σ_v^2 * I),  vol_data[i])
            
#         end
#         # ensemble_prob = EnsembleProblem(prob, 
#         #                                 prob_func=make_prob_func(inits, ρ_t, α_a, α_t, β, η, times), 
#         #                                 output_func=output_func)
        
#         # _esol = solve(ensemble_prob,
#         #                 Tsit5(),
#         # 	            verbose=false,
#         #                 abstol = 1e-6, 
#         #                 reltol = 1e-6, 
#         #                 trajectories=n, 
#         #                 sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    
#         # if !success_condition(get_retcodes(_esol))
#         #     Turing.@addlogprob! -Inf
#         #     println(findall(x -> x == 0, get_retcodes(_esol)))
#         #     println("failed")
#         #     return nothing
#         # end
#         # ab_preds, tau_preds, vol_preds =  split_sols(_esol, ab_tidx, tau_tidx)
        
#         # ab_data ~ MvNormal(ab_preds, σ_a^2 * I)
#         # tau_data ~ MvNormal(tau_preds, σ_a^2 * I)
#         # vol_data ~ MvNormal(vol_preds, σ_a^2 * I) 
#     end