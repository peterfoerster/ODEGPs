import Pkg
Pkg.activate(".")

using ODEGaussianProcesses
using LaTeXStrings
using Plots

import DataInterpolations as DI
import DelimitedFiles as DF
import Integrals
import LinearAlgebra as LA
import Statistics

unicodeplots()
# plotly()

directory = "./csv/vdpo/"

if ~isdir(directory)
    mkdir(directory)
end

directory_ode = directory*"ode/"

if ~isdir(directory_ode)
    mkdir(directory_ode)
end

directory_rep = directory*"reparameterization/"

if ~isdir(directory_rep)
    mkdir(directory_rep)
end

directory_doe = directory*"doe/"

if ~isdir(directory_doe)
    mkdir(directory_doe)
end

# max(|f|) ~ 2.7, mean(|f|) ~ 1.1, I(|f|)/t_f ~ 1.2
# max(|ft|) ~ 0.94, mean(|ft|) ~ 0.56, I(|ft|)/t_f ~ 0.66
# t_f       = 20
# mu        = 1
# (t, x, f) = S_vdpo(0, t_f, mu=mu, abstol=1e-10, reltol=1e-5, directory=directory_ode)
# @info "max(|f|) = $(maximum(abs.(f)))"
# @info "mean(|f|) = $(Statistics.mean(abs.(f)))"
# I = Integrals.SampledIntegralProblem(abs.(f), t)
# S = Integrals.solve(I, Integrals.TrapezoidalRule())
# @info "I(|f|) / t_f = $(S.u/t_f)"
# ft = f ./ sqrt.(1 .+ f.^2)
# @info "max(|ft|) = $(maximum(abs.(ft)))"
# @info "mean(|ft|) = $(Statistics.mean(abs.(ft)))"
# I = Integrals.SampledIntegralProblem(abs.(ft), t)
# S = Integrals.solve(I, Integrals.TrapezoidalRule())
# @info "I(|ft|) / t_f = $(S.u/t_f)"

# max(|f|) ~ 14.2, mean(|f|) ~ 1.9, I(|f|)/t_f ~ 0.4
# max(|ft|) ~ 1, mean(|ft|) ~ 0.37, I(|ft|)/t_f ~ 0.17
# t_f       = 50
# mu        = 10
# (t, x, f) = S_vdpo(0, t_f, mu=mu, abstol=1e-10, reltol=1e-5, directory=directory_ode)
# @info "max(|f|) = $(maximum(abs.(f)))"
# @info "mean(|f|) = $(Statistics.mean(abs.(f)))"
# I = Integrals.SampledIntegralProblem(abs.(f), t)
# S = Integrals.solve(I, Integrals.TrapezoidalRule())
# @info "I(|f|) / t_f = $(S.u/t_f)"
# ft = f ./ sqrt.(1 .+ f.^2)
# @info "max(|ft|) = $(maximum(abs.(ft)))"
# @info "mean(|ft|) = $(Statistics.mean(abs.(ft)))"
# I = Integrals.SampledIntegralProblem(abs.(ft), t)
# S = Integrals.solve(I, Integrals.TrapezoidalRule())
# @info "I(|ft|) / t_f = $(S.u/t_f)"

function vdpo_conventional_doe(t_f::Real, mu::Real, nu_k::Real, sigma::AbstractVector{<:Real}, normalize::Bool, standardize::Bool, reltol_x::Real, reltol_k::Real, derivative::Bool, nu::Int, r_lml::Real, filename::String)
    S  = p -> S_vdpo(0, t_f, mu=mu, abstol=1e-10, reltol=1e-5, directory=directory_ode)
    k  = MaternCovariance(nu_k, sigma=sigma, normalize=normalize, standardize=standardize)
    lb = zeros(0)
    ub = zeros(0)

    @time D = conventional_doe!(k, S, lb, ub, maxiters=200, reltol_x=reltol_x, reltol_k=reltol_k, derivative=derivative, nu=nu, r_lml=r_lml, directory=directory_doe, filename=filename)

    display(plot(range(length(D.x)-length(D.e_x), length(D.x)-1), D.e_x, label="", xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    return nothing
end

# 20 s
m           = 1
t_f         = 20
mu          = 1
nu_k        = Inf
sigma       = [1, 1e-6]
normalize   = false
standardize = true
reltol_x    = 1e-5
reltol_k    = 1e-5
derivative  = false
nu          = 2
r_lml       = 0.5
filename    = "conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe(t_f, mu, nu_k, sigma, normalize, standardize, reltol_x, reltol_k, derivative, nu, r_lml, filename)

# does much worse (31 s)
m          = 1
derivative = true
filename   = "conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe(t_f, mu, nu_k, sigma, normalize, standardize, reltol_x, reltol_k, derivative, nu, r_lml, filename)

# 271 s
m          = 1
nu_k       = 2
derivative = false
r_lml      = 0.5
filename   = "conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe(t_f, mu, nu_k, sigma, normalize, standardize, reltol_x, reltol_k, derivative, nu, r_lml, filename)

# does worse (294 s)
m          = 1
derivative = true
filename   = "conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe(t_f, mu, nu_k, sigma, normalize, standardize, reltol_x, reltol_k, derivative, nu, r_lml, filename)

# 11 s
m          = 1
nu_k       = 1
derivative = false
nu         = 1
r_lml      = 0.5
filename   = "conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe(t_f, mu, nu_k, sigma, normalize, standardize, reltol_x, reltol_k, derivative, nu, r_lml, filename)

# does the same? (13 s)
m          = 1
derivative = true
filename   = "conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe(t_f, mu, nu_k, sigma, normalize, standardize, reltol_x, reltol_k, derivative, nu, r_lml, filename)

# 18 s
m          = 1
nu_k       = 0
derivative = false
nu         = 0
r_lml      = 0.7
filename   = "conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe(t_f, mu, nu_k, sigma, normalize, standardize, reltol_x, reltol_k, derivative, nu, r_lml, filename)

# does the same? (5 s)
m        = 1
nu       = 2
filename = "conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe(t_f, mu, nu_k, sigma, normalize, standardize, reltol_x, reltol_k, derivative, nu, r_lml, filename)

# 10 s
m        = 1
t_f      = 50
mu       = 10
nu_k     = Inf
r_lml    = 0.5
filename = "conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe(t_f, mu, nu_k, sigma, normalize, standardize, reltol_x, reltol_k, derivative, nu, r_lml, filename)

# 9 s
m        = 1
nu_k     = 2
r_lml    = 0.5
filename = "conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe(t_f, mu, nu_k, sigma, normalize, standardize, reltol_x, reltol_k, derivative, nu, r_lml, filename)

# 6 s
m          = 1
nu_k       = 1
derivative = true
nu         = 1
r_lml      = 0.5
filename   = "conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe(t_f, mu, nu_k, sigma, normalize, standardize, reltol_x, reltol_k, derivative, nu, r_lml, filename)

# 23 s
m          = 1
nu_k       = 0
derivative = false
nu         = 0
r_lml      = 0.7
filename   = "conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe(t_f, mu, nu_k, sigma, normalize, standardize, reltol_x, reltol_k, derivative, nu, r_lml, filename)

function vdpo_conventional_least_squares(filename::String, I::AbstractVector{Int})
    e_x = DF.readdlm(filename*"_e_x.csv")[:]
    N_x = length(DF.readdlm(filename*"_x.csv")[:])
    N   = range(N_x-length(e_x), N_x-1)

    # least squares estimate
    A = [ones(length(I)) log.(N[I])]
    x = (A' * A) \ (A' * log.(e_x[I]))
    a = exp(x[1])
    b = x[2]
    r = N -> a * N^b

    display(plot(N, [e_x r.(N)], label=[L"e_x" "LS estimate"], xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    open(filename*"_e_l2.csv", "w") do io
        write(io, "N,e_x\n")
        DF.writedlm(io, [N e_x], ',')
    end

    open(filename*"_ls.csv", "w") do io
        write(io, "a,b\n")
        DF.writedlm(io, [a b], ',')
    end

    return nothing
end

m          = 1
mu         = 1
nu_k       = Inf
derivative = false
nu         = 2
filename   = directory_doe*"conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 185)
# vdpo_conventional_least_squares(filename, I)

m        = 1
nu_k     = 2
filename = directory_doe*"conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 197)
# vdpo_conventional_least_squares(filename, I)

m          = 1
nu_k       = 1
derivative = true
nu         = 1
filename   = directory_doe*"conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 197)
# vdpo_conventional_least_squares(filename, I)

m          = 1
nu_k       = 0
derivative = false
nu         = 0
filename   = directory_doe*"conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 197)
# vdpo_conventional_least_squares(filename, I)

m          = 1
mu         = 10
nu_k       = Inf
nu         = 2
filename   = directory_doe*"conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 197)
# vdpo_conventional_least_squares(filename, I)

m        = 1
nu_k     = 2
filename = directory_doe*"conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 197)
# vdpo_conventional_least_squares(filename, I)

m          = 1
nu_k       = 1
derivative = true
nu         = 1
filename   = directory_doe*"conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 197)
# vdpo_conventional_least_squares(filename, I)

m          = 1
nu_k       = 0
derivative = false
nu         = 0
filename   = directory_doe*"conventional_mu=$(mu)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 197)
# vdpo_conventional_least_squares(filename, I)

function vdpo_reparameterization(t_f::Real, mu_p::Real, n_s::Int, derivative::Bool, nu::Int, mu::Real, filename::String)
    s         = collect(range(0, 1, length=n_s))
    (t, x, f) = S_vdpo(0, t_f, mu=mu_p, abstol=1e-10, reltol=1e-5, directory=directory_ode)
    x_min     = minimum(x)
    x_max     = maximum(x)

    if derivative
        (tauit, xt, tauitp, s_ext) = reparameterize(t, x, f, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7)
    else
        (tauit, xt, tauitp, s_ext) = reparameterize(t, x, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7)
    end

    plot(t / t[end], x, label=L"x(t_\mr{f} s)", xlabel=L"s", linewidth=1.5, legendfontsize=12)
    plot!(s, (x_max - x_min) * (tauit(s) / tauit(1)), label=L"\propto \tilde{\tau}^{-1}(s)", linewidth=1.5)
    display(plot!(s, xt(s), label=L"\tilde{x}(s)", linewidth=1.5, legend=:topleft))
    display(scatter!(s_ext[2:end-1], xt(s_ext[2:end-1]), label=L"(s_{\mr{ext}_i}, \tilde{x}(s_{\mr{ext}_i}))"))
    display(plot(s, tauitp.(s), label="", xlabel=L"s", ylabel=L"\frac{d}{ds} \tilde{\tau}^{-1}(s)", linewidth=1.5, legendfontsize=12))

    open(filename*".csv", "w") do io
        write(io, "s,tauit,xt,tauitp\n")
        DF.writedlm(io, [s tauit(s) xt(s) tauitp.(s)], ',')
    end

    open(filename*"_ext.csv", "w") do io
        write(io, "s,xt\n")
        DF.writedlm(io, [s_ext[2:end-1] xt(s_ext[2:end-1])], ',')
    end

    return nothing
end

t_f        = 20
mu_p       = 1
n_s        = 1000
derivative = false
nu         = 2
mu         = Inf
filename   = directory_rep*"t_f=$(t_f)_mu_p=$(mu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)"
# vdpo_reparameterization(t_f, mu_p, n_s, derivative, nu, mu, filename)

derivative = true
nu         = 1
mu         = 1
filename   = directory_rep*"t_f=$(t_f)_mu_p=$(mu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)"
# vdpo_reparameterization(t_f, mu_p, n_s, derivative, nu, mu, filename)

derivative = false
nu         = 2
mu         = 2
filename   = directory_rep*"t_f=$(t_f)_mu_p=$(mu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)"
# vdpo_reparameterization(t_f, mu_p, n_s, derivative, nu, mu, filename)

t_f      = 50
mu_p     = 10
mu       = Inf
filename = directory_rep*"t_f=$(t_f)_mu_p=$(mu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)"
# vdpo_reparameterization(t_f, mu_p, n_s, derivative, nu, mu, filename)

derivative = true
nu         = 1
mu         = 1
filename   = directory_rep*"t_f=$(t_f)_mu_p=$(mu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)"
# vdpo_reparameterization(t_f, mu_p, n_s, derivative, nu, mu, filename)

derivative = false
nu         = 2
mu         = 2
filename   = directory_rep*"t_f=$(t_f)_mu_p=$(mu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)"
# vdpo_reparameterization(t_f, mu_p, n_s, derivative, nu, mu, filename)

function vdpo_reference_2D(t_f::Real, mu_min::Real, mu_max::Real, n_mu::Int, n_t::Int, filename::String)
    muh = collect(range(mu_min, mu_max, length=n_mu))
    th  = collect(range(0, t_f, length=n_t))
    Ph  = zeros(2, 0)
    xh  = zeros(0)

    for i in range(1, n_mu)
        Ph        = [Ph [th'; repeat(muh[i:i], 1, n_t)]]
        (t, x, f) = S_vdpo(0, t_f, mu=muh[i], abstol=1e-10, reltol=1e-5, directory=directory_ode)
        xh_i      = DI.CubicHermiteSpline(f, x, t)(th)
        append!(xh, xh_i)
    end

    display(heatmap(reshape(xh, n_t, n_mu)', xlabel=L"t", ylabel=L"\mu", colorbar=:none))

    open(filename*".csv", "w") do io
        write(io, "t,mu,x\n")
        DF.writedlm(io, [Ph[1,:] Ph[2,:] xh], ',')
    end

    return nothing
end

t_f      = 20
mu_min   = 0.5
mu_max   = 1.5
n_mu     = 50
n_t      = 200
filename = directory_rep*"t_f=$(t_f)_mu_min=$(mu_min)_mu_max=$(mu_max)_n_mu=$(n_mu)_n_t=$(n_t)_reference"
# vdpo_reference_2D(t_f, mu_min, mu_max, n_mu, n_t, filename)

t_f      = 50
mu_min   = 9.5
mu_max   = 10.5
filename = directory_rep*"t_f=$(t_f)_mu_min=$(mu_min)_mu_max=$(mu_max)_n_mu=$(n_mu)_n_t=$(n_t)_reference"
# vdpo_reference_2D(t_f, mu_min, mu_max, n_mu, n_t, filename)

function vdpo_reparameterization_2D(t_f::Real, mu_min::Real, mu_max::Real, n_mu::Int, n_s::Int, derivative::Bool, nu::Int, mu::Real, filename::String)
    muh     = collect(range(mu_min, mu_max, length=n_mu))
    sh      = collect(range(0, 1, length=n_s))
    Ph      = zeros(2, 0)
    tauith  = zeros(0)
    xth     = zeros(0)
    tauitph = zeros(0)

    for i in range(1, n_mu)
        Ph        = [Ph [sh'; repeat(muh[i:i], 1, n_s)]]
        (t, x, f) = S_vdpo(0, t_f, mu=muh[i], abstol=1e-10, reltol=1e-5, directory=directory_ode)

        if derivative
            (tauit, xt, tauitp, s_ext) = reparameterize(t, x, f, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7)
        else
            (tauit, xt, tauitp, s_ext) = reparameterize(t, x, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7)
        end

        append!(tauith, tauit(sh))
        append!(xth, xt(sh))
        append!(tauitph, tauitp.(sh))
    end

    display(heatmap(reshape(tauith, n_s, n_mu)', xlabel=L"s", ylabel=L"\mu", title=L"\tilde{\tau}^{-1}", colorbar=:none))
    display(heatmap(reshape(xth, n_s, n_mu)', xlabel=L"s", ylabel=L"\mu", title=L"\tilde{x}", colorbar=:none))
    display(heatmap(reshape(tauitph, n_s, n_mu)', xlabel=L"s", ylabel=L"\mu", title=L"\frac{d}{ds} \tilde{\tau}^{-1}", colorbar=:none))

    open(filename*".csv", "w") do io
        write(io, "s,mu,tauit,xt,tauitp\n")
        DF.writedlm(io, [Ph[1,:] Ph[2,:] tauith xth tauitph], ',')
    end

    return nothing
end

t_f        = 20
mu_min     = 0.5
mu_max     = 1.5
n_mu       = 50
n_s        = 200
derivative = false
nu         = 2
mu         = Inf
filename   = directory_rep*"t_f=$(t_f)_mu_min=$(mu_min)_mu_max=$(mu_max)_n_mu=$(n_mu)_n_s=$(n_s)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)"
# vdpo_reparameterization_2D(t_f, mu_min, mu_max, n_mu, n_s, derivative, nu, mu, filename)

t_f      = 50
mu_min   = 9.5
mu_max   = 10.5
filename = directory_rep*"t_f=$(t_f)_mu_min=$(mu_min)_mu_max=$(mu_max)_n_mu=$(n_mu)_n_s=$(n_s)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)"
# vdpo_reparameterization_2D(t_f, mu_min, mu_max, n_mu, n_s, derivative, nu, mu, filename)

derivative = true
nu         = 1
mu         = 1
filename   = directory_rep*"t_f=$(t_f)_mu_min=$(mu_min)_mu_max=$(mu_max)_n_mu=$(n_mu)_n_s=$(n_s)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)"
# vdpo_reparameterization_2D(t_f, mu_min, mu_max, n_mu, n_s, derivative, nu, mu, filename)

derivative = false
nu         = 2
mu         = 2
filename   = directory_rep*"t_f=$(t_f)_mu_min=$(mu_min)_mu_max=$(mu_max)_n_mu=$(n_mu)_n_s=$(n_s)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)"
# vdpo_reparameterization_2D(t_f, mu_min, mu_max, n_mu, n_s, derivative, nu, mu, filename)

function vdpo_reparameterized_doe(t_f::Real, mu_p::Real, nu_k::Real, sigma::AbstractVector{<:Real}, normalize::Bool, standardize::Bool, reltol_v::Real, reltol_k::Real, derivative::Bool, nu::Int, mu::Real, r_lml::Real, filename::String)
    S  = p -> S_vdpo(0, t_f, mu=mu_p, abstol=1e-10, reltol=1e-5, directory=directory_ode)
    k  = MaternCovariance(nu_k, sigma=sigma, normalize=normalize, standardize=standardize)
    lb = zeros(0)
    ub = zeros(0)

    @time D_tauit = reparameterized_doe!(k, S, lb, ub, variable="tauit", maxiters=200, reltol_v=reltol_v, reltol_k=reltol_k, derivative=derivative, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7, r_lml=r_lml, directory=directory_doe, filename=filename*"_tauit")

    display(plot(range(length(D_tauit.tauit)-length(D_tauit.e_v), length(D_tauit.tauit)-1), D_tauit.e_v, label=L"e_{\tilde{\tau}^{-1}}", xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    @time D_xt = reparameterized_doe!(k, S, lb, ub, variable="xt", maxiters=200, reltol_v=reltol_v, reltol_k=reltol_k, derivative=derivative, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7, r_lml=r_lml, directory=directory_doe, filename=filename*"_xt")

    display(plot(range(length(D_xt.xt)-length(D_xt.e_v), length(D_xt.xt)-1), D_xt.e_v, label=L"e_{\tilde{x}}", xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    return nothing
end

# 25/25 s, 13/30 s
m           = 2
t_f         = 20
mu_p        = 1
nu_k        = Inf
sigma       = [1, 1e-6]
normalize   = false
standardize = true
reltol_v    = 1e-5
reltol_k    = 1e-5
derivative  = false
nu          = 2
mu          = Inf
r_lml       = 0.5
filename    = "reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe(t_f, mu_p, nu_k, sigma, normalize, standardize, reltol_v, reltol_k, derivative, nu, mu, r_lml, filename)

# 7/10 s
m        = 1
nu_k     = 0
r_lml    = 0.5
filename = "reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe(t_f, mu_p, nu_k, sigma, normalize, standardize, reltol_v, reltol_k, derivative, nu, mu, r_lml, filename)

# 15/15 s, 9/10 s
m          = 2
nu_k       = 1
derivative = true
nu         = 1
mu         = 1
r_lml      = 0.5
filename   = "reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe(t_f, mu_p, nu_k, sigma, normalize, standardize, reltol_v, reltol_k, derivative, nu, mu, r_lml, filename)

# 32/17 s
m        = 1
nu_k     = Inf
r_lml    = 0.5
filename = "reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe(t_f, mu_p, nu_k, sigma, normalize, standardize, reltol_v, reltol_k, derivative, nu, mu, r_lml, filename)

# 23/8 s, 17/13 s
m          = 2
nu_k       = 2
derivative = false
nu         = 2
mu         = 2
r_lml      = 0.5
filename   = "reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe(t_f, mu_p, nu_k, sigma, normalize, standardize, reltol_v, reltol_k, derivative, nu, mu, r_lml, filename)

# 18/15 s
m        = 1
nu_k     = Inf
r_lml    = 0.5
filename = "reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe(t_f, mu_p, nu_k, sigma, normalize, standardize, reltol_v, reltol_k, derivative, nu, mu, r_lml, filename)

# 21/18 s, 19/19 s
m        = 2
t_f      = 50
mu_p     = 10
mu       = Inf
r_lml    = 0.5
filename = "reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe(t_f, mu_p, nu_k, sigma, normalize, standardize, reltol_v, reltol_k, derivative, nu, mu, r_lml, filename)

# 6/9 s
m        = 1
nu_k     = 0
r_lml    = 0.5
filename = "reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe(t_f, mu_p, nu_k, sigma, normalize, standardize, reltol_v, reltol_k, derivative, nu, mu, r_lml, filename)

# 13/12 s
m          = 1
nu_k       = 1
derivative = true
nu         = 1
mu         = 1
r_lml      = 0.5
filename   = "reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe(t_f, mu_p, nu_k, sigma, normalize, standardize, reltol_v, reltol_k, derivative, nu, mu, r_lml, filename)

# 21/25 s
m        = 1
nu_k     = Inf
r_lml    = 0.5
filename = "reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe(t_f, mu_p, nu_k, sigma, normalize, standardize, reltol_v, reltol_k, derivative, nu, mu, r_lml, filename)

# 12/9 s
m          = 1
nu_k       = 2
derivative = false
nu         = 2
mu         = 2
r_lml      = 0.5
filename   = "reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe(t_f, mu_p, nu_k, sigma, normalize, standardize, reltol_v, reltol_k, derivative, nu, mu, r_lml, filename)

# 23/17 s
m        = 1
nu_k     = Inf
r_lml    = 0.5
filename = "reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe(t_f, mu_p, nu_k, sigma, normalize, standardize, reltol_v, reltol_k, derivative, nu, mu, r_lml, filename)

function vdpo_reparameterized_least_squares(filename::String, I_tauit::AbstractVector{Int}, I_xt::AbstractVector{Int})
    e_tauit = DF.readdlm(filename*"_tauit_e_tauit.csv")[:]
    N_tauit = length(DF.readdlm(filename*"_tauit_tauit.csv")[:])
    N_tauit = range(N_tauit-length(e_tauit), N_tauit-1)
    e_xt    = DF.readdlm(filename*"_xt_e_xt.csv")[:]
    N_xt    = length(DF.readdlm(filename*"_xt_xt.csv")[:])
    N_xt    = range(N_xt-length(e_xt), N_xt-1)

    # least squares estimates
    A = [ones(length(I_tauit)) log.(N_tauit[I_tauit])]
    x = (A' * A) \ (A' * log.(e_tauit[I_tauit]))
    a = exp(x[1])
    b = x[2]
    r = N -> a * N^b

    display(plot(N_tauit, [e_tauit r.(N_tauit)], label=[L"e_{\tilde{\tau}^{-1}}" "LS estimate"], xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    open(filename*"_tauit_e_l2.csv", "w") do io
        write(io, "N,e_tauit\n")
        DF.writedlm(io, [N_tauit e_tauit], ',')
    end

    open(filename*"_tauit_ls.csv", "w") do io
        write(io, "a,b\n")
        DF.writedlm(io, [a b], ',')
    end

    A = [ones(length(I_xt)) log.(N_xt[I_xt])]
    x = (A' * A) \ (A' * log.(e_xt[I_xt]))
    a = exp(x[1])
    b = x[2]
    r = N -> a * N^b

    display(plot(N_xt, [e_xt r.(N_xt)], label=[L"e_{\tilde{x}}" "LS estimate"], xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    open(filename*"_xt_e_l2.csv", "w") do io
        write(io, "N,e_xt\n")
        DF.writedlm(io, [N_xt e_xt], ',')
    end

    open(filename*"_xt_ls.csv", "w") do io
        write(io, "a,b\n")
        DF.writedlm(io, [a b], ',')
    end

    return nothing
end

m          = 2
mu_p       = 1
nu_k       = Inf
derivative = false
nu         = 2
mu         = Inf
filename   = directory_doe*"reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 197)
I_xt       = range(10, 197)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

m        = 1
nu_k     = 0
filename = directory_doe*"reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 197)
I_xt     = range(10, 197)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 2
nu_k       = 1
derivative = true
nu         = 1
mu         = 1
filename   = directory_doe*"reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 197)
I_xt       = range(10, 197)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

m        = 1
nu_k     = Inf
filename = directory_doe*"reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 197)
I_xt     = range(10, 197)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 2
nu_k       = 2
derivative = false
nu         = 2
mu         = 2
filename   = directory_doe*"reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 187)
# I_xt       = range(10, 167)
I_xt       = range(10, 197)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

m        = 1
nu_k     = Inf
filename = directory_doe*"reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 177)
I_xt     = range(10, 197)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

m        = 2
mu_p     = 10
mu       = Inf
filename = directory_doe*"reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 197)
I_xt     = range(10, 197)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

m        = 1
nu_k     = 0
filename = directory_doe*"reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 197)
I_xt     = range(10, 197)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 1
nu_k       = 1
derivative = true
nu         = 1
mu         = 1
filename   = directory_doe*"reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 197)
I_xt       = range(10, 197)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

m        = 1
nu_k     = Inf
filename = directory_doe*"reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 177)
I_xt     = range(10, 197)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 1
nu_k       = 2
derivative = false
nu         = 2
mu         = 2
filename   = directory_doe*"reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 197)
I_xt       = range(10, 197)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

m        = 1
nu_k     = Inf
filename = directory_doe*"reparameterized_mu_p=$(mu_p)_nu_k=$(nu_k)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 177)
I_xt     = range(10, 197)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

function vdpo_conventional_doe_2D(t_f::Real, mu_min::Real, mu_max::Real, nu_t::Real, nu_p::Real, sigma::AbstractVector{<:Real}, normalize::Bool, standardize::Bool, reltol_x::Real, reltol_k::Real, abstol_p::Real, derivative::Bool, nu::Int, r_lml::Real, filename::String)
    S  = p -> S_vdpo(0, t_f, mu=p[1], abstol=1e-10, reltol=1e-5, directory=directory_ode)
    k  = MaternCovariance(nu_t, sigma=sigma, normalize=normalize, standardize=standardize) * MaternCovariance(nu_p, normalize=normalize, standardize=standardize)
    lb = [mu_min]
    ub = [mu_max]

    @time D = conventional_doe!(k, S, lb, ub, maxiters=400, reltol_x=reltol_x, reltol_k=reltol_k, abstol_p=[1e-6*t_f; abstol_p*(ub - lb)], derivative=derivative, nu=nu, r_lml=r_lml, n_test_p=10, directory=directory_doe, filename=filename)

    P = unique(D.P[2,:])
    @info "length(P) = $(length(P))"
    scatter(D.P[1,:], D.P[2,:])
    display(scatter!(D.P_e[1,:], D.P_e[2,:]))

    display(plot(range(length(D.x)-length(D.e_x), length(D.x)-1), D.e_x, label="", xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    return nothing
end

# 481 s
m           = 1
t_f         = 20
mu_min      = 0.5
mu_max      = 1.5
nu_t        = Inf
nu_p        = Inf
sigma       = [1, 1e-6]
normalize   = false
standardize = true
reltol_x    = 1e-3
reltol_k    = 1e-3
abstol_p    = 1e-1
derivative  = false
nu          = 2
r_lml       = 0.5
filename    = "conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe_2D(t_f, mu_min, mu_max, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 315 s
m        = 2
abstol_p = 1e-3
filename = "conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe_2D(t_f, mu_min, mu_max, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 124 s
m        = 1
t_f      = 50
mu_min   = 9.5
mu_max   = 10.5
abstol_p = 1e-1
r_lml    = 0.5
filename = "conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe_2D(t_f, mu_min, mu_max, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

filename = directory_doe*"conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 391)
# vdpo_conventional_least_squares(filename, I)

# 123 s
m        = 1
nu_t     = 2
r_lml    = 0.5
filename = "conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe_2D(t_f, mu_min, mu_max, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

filename = directory_doe*"conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 391)
# vdpo_conventional_least_squares(filename, I)

# 109 s
m          = 1
nu_t       = 1
derivative = true
nu         = 1
r_lml      = 0.5
filename   = "conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe_2D(t_f, mu_min, mu_max, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

filename = directory_doe*"conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 391)
# vdpo_conventional_least_squares(filename, I)

# 567 s
m          = 1
nu_t       = 0
derivative = false
nu         = 0
r_lml      = 0.5
filename   = "conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe_2D(t_f, mu_min, mu_max, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

filename = directory_doe*"conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 391)
# vdpo_conventional_least_squares(filename, I)

m        = 1
t_f      = 20
mu_min   = 0.5
mu_max   = 1.5
nu_t     = Inf
nu       = 2
filename = directory_doe*"conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 391)
# vdpo_conventional_least_squares(filename, I)

# 222 s
m        = 1
nu_t     = 2
r_lml    = 0.5
filename = "conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe_2D(t_f, mu_min, mu_max, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

filename = directory_doe*"conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 391)
# vdpo_conventional_least_squares(filename, I)

# 114 s
m          = 1
nu_t       = 1
derivative = true
nu         = 1
r_lml      = 0.5
filename   = "conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe_2D(t_f, mu_min, mu_max, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

filename = directory_doe*"conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 391)
# vdpo_conventional_least_squares(filename, I)

# ? s
m          = 1
nu_t       = 0
derivative = false
nu         = 0
r_lml      = 0.7
filename   = "conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# vdpo_conventional_doe_2D(t_f, mu_min, mu_max, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

filename = directory_doe*"conventional_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 391)
# vdpo_conventional_least_squares(filename, I)

function vdpo_reparameterized_doe_2D(t_f::Real, mu_min::Real, mu_max::Real, nu_s::Real, nu_p::Real, sigma::AbstractVector{<:Real}, normalize::Bool, standardize::Bool, reltol_v::Real, reltol_k::Real, abstol_p::Real, derivative::Bool, nu::Int, mu::Real, r_lml::Real, filename::String)
    S  = p -> S_vdpo(0, t_f, mu=p[1], abstol=1e-10, reltol=1e-5, directory=directory_ode)
    k  = MaternCovariance(nu_s, sigma=sigma, normalize=normalize, standardize=standardize) * MaternCovariance(nu_p, normalize=normalize, standardize=standardize)
    lb = [mu_min]
    ub = [mu_max]

    @time D_tauit = reparameterized_doe!(k, S, lb, ub, variable="tauit", maxiters=400, reltol_v=reltol_v, reltol_k=reltol_k, abstol_p=[1e-6*t_f; abstol_p*(ub - lb)], derivative=derivative, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7, r_lml=r_lml, n_test_p=10, directory=directory_doe, filename=filename*"_tauit")

    P = unique(D_tauit.P[2,:])
    @info "length(P) = $(length(P))"
    scatter(D_tauit.P[1,:], D_tauit.P[2,:])
    display(scatter!(D_tauit.P_e[1,:], D_tauit.P_e[2,:]))

    display(plot(range(length(D_tauit.tauit)-length(D_tauit.e_v), length(D_tauit.tauit)-1), D_tauit.e_v, label=L"e_{\tilde{\tau}^{-1}}", xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    @time D_xt = reparameterized_doe!(k, S, lb, ub, variable="xt", maxiters=400, reltol_v=reltol_v, reltol_k=reltol_k, abstol_p=[1e-6*t_f; abstol_p*(ub - lb)], derivative=derivative, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7, r_lml=r_lml, n_test_p=10, directory=directory_doe, filename=filename*"_xt")

    display(plot(range(length(D_xt.xt)-length(D_xt.e_v), length(D_xt.xt)-1), D_xt.e_v, label=L"e_{\tilde{x}}", xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    return nothing
end

# 333/178 s
m           = 1
t_f         = 50
mu_min      = 9.5
mu_max      = 10.5
nu_s        = Inf
nu_p        = Inf
sigma       = [1, 1e-6]
normalize   = false
standardize = true
reltol_v    = 1e-3
reltol_k    = 1e-3
abstol_p    = 1e-1
derivative  = false
nu          = 2
mu          = Inf
r_lml       = 0.5
filename    = "reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe_2D(t_f, mu_min, mu_max, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

filename = directory_doe*"reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 391)
I_xt     = range(10, 391)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

# 572/202 s
m        = 1
nu_s     = 0
r_lml    = 0.5
filename = "reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe_2D(t_f, mu_min, mu_max, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

filename = directory_doe*"reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 391)
I_xt     = range(10, 391)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

# 328/147 s
m          = 1
nu_s       = 1
derivative = true
nu         = 1
mu         = 1
r_lml      = 0.5
filename   = "reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe_2D(t_f, mu_min, mu_max, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

filename = directory_doe*"reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 391)
I_xt     = range(10, 391)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

# 295/194 s
m          = 1
nu_s       = 2
derivative = false
nu         = 2
mu         = 2
r_lml      = 0.5
filename   = "reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe_2D(t_f, mu_min, mu_max, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

filename = directory_doe*"reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 391)
I_xt     = range(10, 391)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

# 536/183 s
m        = 1
t_f      = 20
mu_min   = 0.5
mu_max   = 1.5
nu_s     = Inf
mu       = Inf
r_lml    = 0.5
filename = "reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe_2D(t_f, mu_min, mu_max, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

filename = directory_doe*"reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 391)
I_xt     = range(10, 391)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

# 445/209 s
m        = 1
nu_s     = 0
r_lml    = 0.5
filename = "reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe_2D(t_f, mu_min, mu_max, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

filename = directory_doe*"reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 391)
I_xt     = range(10, 391)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

# 214/247 s
m          = 1
nu_s       = 1
derivative = true
nu         = 1
mu         = 1
r_lml      = 0.5
filename   = "reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe_2D(t_f, mu_min, mu_max, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

filename = directory_doe*"reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 391)
I_xt     = range(10, 391)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

# 198/219 s, 477/551 s
m          = 1
nu_s       = 2
derivative = false
nu         = 2
mu         = 2
r_lml      = 0.5
filename   = "reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_doe_2D(t_f, mu_min, mu_max, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

filename = directory_doe*"reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 391)
I_xt     = range(10, 391)
# vdpo_reparameterized_least_squares(filename, I_tauit, I_xt)

function vdpo_reparameterized_prediction_2D(t_f::Real, mu_min::Real, mu_max::Real, nu_s::Real, nu_p::Real, normalize::Bool, standardize::Bool, n_mu::Int, n_s::Int, filename::String)
    k   = MaternCovariance(nu_s, normalize=normalize, standardize=standardize) * MaternCovariance(nu_p, normalize=normalize, standardize=standardize)
    muh = collect(range(mu_min, mu_max, length=n_mu))
    sh  = collect(range(0, 1, length=n_s))
    Ph  = zeros(2, 0)

    for i in range(1, n_mu)
        Ph = [Ph [sh'; repeat(muh[i:i], 1, n_s)]]
    end

    btheta  = DF.readdlm(filename*"_tauit_btheta.csv")[:]
    P       = DF.readdlm(filename*"_tauit_P.csv", ',')[:,1:end-1]
    x       = DF.readdlm(filename*"_tauit_tauit.csv")[1:end-1]
    U       = DF.readdlm(filename*"_tauit_U.csv", ',')
    p       = DF.readdlm(filename*"_tauit_p.csv", Int)[:]
    k.sigma = btheta[1:2]
    k.theta = btheta[3:end]
    C       = LA.CholeskyPivoted(LA.UpperTriangular(U), 'U', p, length(p), 1e-16, 0)

    if k.normalize
        (m_P, n_P) = normalize!(P)
        normalize!(Ph, m_P, n_P)
    end

    if k.standardize
        (mu_x, sigma_x) = standardize!(x)
    end

    tauith = posterior(k, P, x, Ph, C=C)

    if k.normalize
        unnormalize!(P, m_P, n_P)
        unnormalize!(Ph, m_P, n_P)
    end

    if k.standardize
        unstandardize!(x, mu_x, sigma_x)
        unstandardize!(tauith, mu_x, sigma_x)
    end

    # ensure tauith matches t on the boundary
    for i in range(1, n_mu)
        tauith[(i-1)*n_s+1] = 0
        tauith[i*n_s]       = t_f
    end

    display(heatmap(reshape(tauith, n_s, n_mu)', xlabel=L"s", ylabel=L"\mu", colorbar=:none))

    btheta  = DF.readdlm(filename*"_xt_btheta.csv")[:]
    P       = DF.readdlm(filename*"_xt_P.csv", ',')[:,1:end-1]
    x       = DF.readdlm(filename*"_xt_xt.csv")[1:end-1]
    U       = DF.readdlm(filename*"_xt_U.csv", ',')
    p       = DF.readdlm(filename*"_xt_p.csv", Int)[:]
    k.sigma = btheta[1:2]
    k.theta = btheta[3:end]
    C       = LA.CholeskyPivoted(LA.UpperTriangular(U), 'U', p, length(p), 1e-16, 0)

    if k.normalize
        (m_P, n_P) = normalize!(P)
        normalize!(Ph, m_P, n_P)
    end

    if k.standardize
        (mu_x, sigma_x) = standardize!(x)
    end

    xth = posterior(k, P, x, Ph, C=C)

    if k.normalize
        unnormalize!(P, m_P, n_P)
        unnormalize!(Ph, m_P, n_P)
    end

    if k.standardize
        unstandardize!(x, mu_x, sigma_x)
        unstandardize!(xth, mu_x, sigma_x)
    end

    display(heatmap(reshape(xth, n_s, n_mu)', xlabel=L"s", ylabel=L"\mu", colorbar=:none))

    open(filename*"_prediction.csv", "w") do io
        write(io, "s,mu,tauith,xth\n")
        DF.writedlm(io, [Ph[1,:] Ph[2,:] tauith xth], ',')
    end

    return nothing
end

m           = 1
t_f         = 50
mu_min      = 9.5
mu_max      = 10.5
nu_s        = 1
nu_p        = Inf
normalize   = false
standardize = true
n_mu        = 50
n_s         = 200
derivative  = true
nu          = 1
mu          = 1
filename    = directory_doe*"reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_prediction_2D(t_f, mu_min, mu_max, nu_s, nu_p, normalize, standardize, n_mu, n_s, filename)

m          = 2
t_f        = 20
mu_min     = 0.5
mu_max     = 1.5
nu_s       = 2
derivative = false
nu         = 2
mu         = 2
filename   = directory_doe*"reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_prediction_2D(t_f, mu_min, mu_max, nu_s, nu_p, normalize, standardize, n_mu, n_s, filename)

m           = 1
nu_s        = 1
derivative  = true
nu          = 1
mu          = 1
filename    = directory_doe*"reparameterized_mu_min=$(mu_min)_mu_max=$(mu_max)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# vdpo_reparameterized_prediction_2D(t_f, mu_min, mu_max, nu_s, nu_p, normalize, standardize, n_mu, n_s, filename)

# computational cost
# t_conventional_2D          = [124, 123, 109, 567]
# t_reparameterized_2D_tauit = [178, 202, 147, 194]
# t_reparameterized_2D_xt    = [333, 572, 328, 295]
# Statistics.mean(t_conventional_2D)
# Statistics.std(t_conventional_2D)
# Statistics.mean(t_reparameterized_2D_tauit)
# Statistics.std(t_reparameterized_2D_tauit)
# Statistics.mean(t_reparameterized_2D_xt)
# Statistics.std(t_reparameterized_2D_xt)
