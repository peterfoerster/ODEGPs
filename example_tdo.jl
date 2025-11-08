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

directory = "./csv/tdo/"

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

# first component
# max(|f|) ~ 1e5, mean(|f|) ~ 7.4e3, I(|f|)/t_f ~ 214
# max(|ft|) ~ 1, mean(|ft|) ~ 1, I(|ft|)/t_f ~ 1
# second component
# max(|f|) ~ 132, mean(|f|) ~ 56, I(|f|)/t_f ~ 50
# max(|ft|) ~ 1, mean(|ft|) ~ 1, I(|ft|)/t_f ~ 1
# t_f       = 1e-2
# C         = 1e-6
# L         = 3e-3
# i_x       = 2
# (t, x, f) = S_tdo(0, t_f, L=L, C=C, i_x=i_x, abstol=1e-10, reltol=1e-5, directory=directory_ode)
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

function tdo_reference_3D(t_f::Real, C_min::Real, C_max::Real, C::Real, L_min::Real, L_max::Real, L::Real, n_C::Int, n_L::Int, n_t::Int, i_x::Int, filename::String)
    Ch = collect(10 .^ range(log10(C_min), log10(C_max), length=n_C))
    Lh = collect(range(L_min, L_max, length=n_L))
    th = collect(range(0, t_f, length=n_t))
    Ph = zeros(2, 0)
    xh = zeros(0)

    for i in range(1, n_C)
        Ph        = [Ph [th'; repeat(Ch[i:i], 1, n_t)]]
        (t, x, f) = S_tdo(0, t_f, L=L, C=Ch[i], i_x=i_x, abstol=1e-10, reltol=1e-5, directory=directory_ode)
        xh_i      = DI.CubicHermiteSpline(f, x, t)(th)
        append!(xh, xh_i)
    end

    @info "max(xh) = $(maximum(xh))"
    display(heatmap(reshape(xh, n_t, n_C)', xlabel=L"t", ylabel=L"C", colorbar=:none))

    open(filename*"_C_min=$(C_min)_C_max=$(C_max).csv", "w") do io
        write(io, "t,C,x\n")
        DF.writedlm(io, [Ph[1,:] Ph[2,:] xh], ',')
    end

    Ph = zeros(2, 0)
    xh = zeros(0)

    for i in range(1, n_L)
        Ph = [Ph [th'; repeat(Lh[i:i], 1, n_t)]]

        (t, x, f) = S_tdo(0, t_f, L=Lh[i], C=C, i_x=i_x, abstol=1e-10, reltol=1e-5, directory=directory_ode)

        xh_i = DI.CubicHermiteSpline(f, x, t)(th)
        append!(xh, xh_i)
    end

    @info "max(xh) = $(maximum(xh))"
    display(heatmap(reshape(xh, n_t, n_L)', xlabel=L"t", ylabel=L"L", colorbar=:none))

    open(filename*"_L_min=$(L_min)_L_max=$(L_max).csv", "w") do io
        write(io, "t,L,x\n")
        DF.writedlm(io, [Ph[1,:] Ph[2,:] xh], ',')
    end

    return nothing
end

# max(xh) ~ 0.54
t_f      = 1e-2
C_min    = 0.1e-6
C_max    = 10e-6
C        = 1e-6
L_min    = 2.9e-3
L_max    = 3.1e-3
L        = 3e-3
n_C      = 50
n_L      = 50
n_t      = 200
i_x      = 1
filename = directory_rep*"t_f=$(t_f)_C_min=$(C_min)_C_max=$(C_max)_C=$(C)_L_min=$(L_min)_L_max=$(L_max)_L=$(L)_n_C=$(n_C)_n_L=$(n_L)_n_t=$(n_t)_i_x=$(i_x)_reference"
# tdo_reference_3D(t_f, C_min, C_max, C, L_min, L_max, L, n_C, n_L, n_t, i_x, filename)

# max(xh) ~ 0.12
i_x      = 2
filename = directory_rep*"t_f=$(t_f)_C_min=$(C_min)_C_max=$(C_max)_C=$(C)_L_min=$(L_min)_L_max=$(L_max)_L=$(L)_n_C=$(n_C)_n_L=$(n_L)_n_t=$(n_t)_i_x=$(i_x)_reference"
# tdo_reference_3D(t_f, C_min, C_max, C, L_min, L_max, L, n_C, n_L, n_t, i_x, filename)

function tdo_conventional_doe_3D(t_f::Real, C_min::Real, C_max::Real, L_min::Real, L_max::Real, i_x::Int, nu_t::Real, nu_p::Real, sigma::AbstractVector{<:Real}, normalize::Bool, standardize::Bool, reltol_x::Real, reltol_k::Real, abstol_p::Real, derivative::Bool, nu::Int, r_lml::Real, filename::String)
    S  = p -> S_tdo(0, t_f, L=p[2], C=p[1], i_x=i_x, abstol=1e-10, reltol=1e-5, directory=directory_ode)
    k  = MaternCovariance(nu_t, sigma=sigma, normalize=normalize, standardize=standardize) * MaternCovariance(nu_p, 2, normalize=normalize, standardize=standardize)
    lb = [C_min, L_min]
    ub = [C_max, L_max]

    @time D = conventional_doe!(k, S, lb, ub, maxiters=800, reltol_x=reltol_x, reltol_k=reltol_k, abstol_p=[1e-6*t_f; abstol_p*(ub - lb)], derivative=derivative, nu=nu, r_lml=r_lml, n_test_p=10, directory=directory_doe, filename=filename)

    P = unique(D.P[2:3,:], dims=2)
    @info "length(P) = $(size(P, 2))"
    scatter(P[1,:], P[2,:])
    display(scatter!(D.P_e[2,:], D.P_e[3,:]))

    display(plot(range(length(D.x)-length(D.e_x), length(D.x)-1), D.e_x, label="", xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    return nothing
end

# 632 s, 167 combinations
m           = 1
t_f         = 1e-2
C_min       = 0.1e-6
C_max       = 10e-6
L_min       = 2.9e-3
L_max       = 3.1e-3
i_x         = 2
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
filename    = "conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# tdo_conventional_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 530 s, 136 combinations
m        = 1
i_x      = 1
r_lml    = 0.5
filename = "conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# tdo_conventional_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 567 s, 130 combinations
m        = 1
nu_t     = 2
r_lml    = 0.5
filename = "conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# tdo_conventional_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 671 s, 147 combinations
m          = 1
nu_t       = 1
derivative = true
nu         = 1
r_lml      = 0.5
filename   = "conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# tdo_conventional_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 268 s, 309 combinations
m          = 1
nu_t       = 0
derivative = false
nu         = 0
r_lml      = 0.5
filename   = "conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# tdo_conventional_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 540 s
m        = 1
i_x      = 2
nu_t     = 2
nu       = 2
r_lml    = 0.5
filename = "conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# tdo_conventional_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 833 s
m          = 1
nu_t       = 1
derivative = true
nu         = 1
r_lml      = 0.7
filename   = "conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# tdo_conventional_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 361 s
m          = 1
nu_t       = 0
derivative = false
nu         = 0
r_lml      = 0.5
filename   = "conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# tdo_conventional_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

function tdo_conventional_least_squares(filename::String, I::AbstractVector{Int})
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
C_min      = 0.1e-6
C_max      = 10e-6
L_min      = 2.9e-3
L_max      = 3.1e-3
i_x        = 1
nu_t       = Inf
nu_p       = Inf
derivative = false
nu         = 2
filename   = directory_doe*"conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 785)
# tdo_conventional_least_squares(filename, I)

m        = 1
nu_t     = 2
filename = directory_doe*"conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 785)
# tdo_conventional_least_squares(filename, I)

m          = 1
nu_t       = 1
derivative = true
nu         = 1
filename   = directory_doe*"conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 785)
# tdo_conventional_least_squares(filename, I)

m          = 1
nu_t       = 0
derivative = false
nu         = 0
filename   = directory_doe*"conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 785)
# tdo_conventional_least_squares(filename, I)

m        = 1
i_x      = 2
nu_t     = Inf
nu       = 2
filename = directory_doe*"conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 785)
# tdo_conventional_least_squares(filename, I)

m        = 1
nu_t     = 2
filename = directory_doe*"conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 785)
# tdo_conventional_least_squares(filename, I)

m          = 1
nu_t       = 1
derivative = true
nu         = 1
filename   = directory_doe*"conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 785)
# tdo_conventional_least_squares(filename, I)

m          = 1
nu_t       = 0
derivative = false
nu         = 0
filename   = directory_doe*"conventional_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 785)
# tdo_conventional_least_squares(filename, I)

function tdo_reparameterized_doe_3D(t_f::Real, C_min::Real, C_max::Real, L_min::Real, L_max::Real, i_x::Int, nu_s::Real, nu_p::Real, sigma::AbstractVector{<:Real}, normalize::Bool, standardize::Bool, reltol_v::Real, reltol_k::Real, abstol_p::Real, derivative::Bool, nu::Int, mu::Real, r_lml::Real, filename::String)
    S  = p -> S_tdo(0, t_f, L=p[2], C=p[1], i_x=i_x, abstol=1e-10, reltol=1e-5, directory=directory_ode)
    k  = MaternCovariance(nu_s, sigma=sigma, normalize=normalize, standardize=standardize) * MaternCovariance(nu_p, 2, normalize=normalize, standardize=standardize)
    lb = [C_min, L_min]
    ub = [C_max, L_max]

    @time D_tauit = reparameterized_doe!(k, S, lb, ub, variable="tauit", maxiters=800, reltol_v=reltol_v, reltol_k=reltol_k, abstol_p=[1e-6*t_f; abstol_p*(ub - lb)], derivative=derivative, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7, r_lml=r_lml, n_test_p=10, directory=directory_doe, filename=filename*"_tauit")

    P = unique(D_tauit.P[2:3,:], dims=2)
    @info "length(P) = $(size(P, 2))"
    scatter(P[1,:], P[2,:])
    display(scatter!(D_tauit.P_e[2,:], D_tauit.P_e[3,:]))

    display(plot(range(length(D_tauit.tauit)-length(D_tauit.e_v), length(D_tauit.tauit)-1), D_tauit.e_v, label=L"e_{\tilde{\tau}^{-1}}", xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    @time D_xt = reparameterized_doe!(k, S, lb, ub, variable="xt", maxiters=800, reltol_v=reltol_v, reltol_k=reltol_k, abstol_p=[1e-6*t_f; abstol_p*(ub - lb)], derivative=derivative, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7, r_lml=r_lml, n_test_p=10, directory=directory_doe, filename=filename*"_xt")

    P = unique(D_xt.P[2:3,:], dims=2)
    @info "length(P) = $(size(P, 2))"
    scatter(P[1,:], P[2,:])
    display(scatter!(D_xt.P_e[2,:], D_xt.P_e[3,:]))

    display(plot(range(length(D_xt.xt)-length(D_xt.e_v), length(D_xt.xt)-1), D_xt.e_v, label=L"e_{\tilde{x}}", xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    return nothing
end

# 572/586 s, 110/131 combinations
m           = 1
t_f         = 1e-2
C_min       = 0.1e-6
C_max       = 10e-6
L_min       = 2.9e-3
L_max       = 3.1e-3
i_x         = 1
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
filename    = "reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# tdo_reparameterized_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# 419/472+ s, 19/16 combinations
m        = 1
nu_s     = 0
r_lml    = 0.7
filename = "reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# tdo_reparameterized_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# 504+/402 s, 135/48 combinations
m          = 1
nu_s       = 1
derivative = true
nu         = 1
mu         = 1
r_lml      = 1
filename   = "reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# tdo_reparameterized_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# 784/801 s, 172/181 combinations
m          = 1
nu_s       = 2
derivative = false
nu         = 2
mu         = 2
r_lml      = 0.5
filename   = "reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# tdo_reparameterized_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# ?/398 s
m           = 1
i_x         = 2
nu_s        = Inf
mu          = Inf
r_lml       = 0.7
filename    = "reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# tdo_reparameterized_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# ?/173 s
m        = 1
nu_s     = 0
r_lml    = 0.7
filename = "reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# tdo_reparameterized_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# 497/534 s, 571/504 s
m          = 2
nu_s       = 1
derivative = true
nu         = 1
mu         = 1
r_lml      = 0.5
filename   = "reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# tdo_reparameterized_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# 573/649 s
m          = 1
nu_s       = 2
derivative = false
nu         = 2
mu         = 2
r_lml      = 0.7
filename   = "reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# tdo_reparameterized_doe_3D(t_f, C_min, C_max, L_min, L_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

function tdo_reparameterized_least_squares(filename::String, I_tauit::AbstractVector{Int}, I_xt::AbstractVector{Int})
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

m          = 1
C_min      = 0.1e-6
C_max      = 10e-6
L_min      = 2.9e-3
L_max      = 3.1e-3
i_x        = 1
nu_s       = Inf
nu_p       = Inf
derivative = false
nu         = 2
mu         = Inf
filename   = directory_doe*"reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 785)
I_xt       = range(10, 785)
# tdo_reparameterized_least_squares(filename, I_tauit, I_xt)

m        = 1
nu_s     = 0
filename = directory_doe*"reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 785)
I_xt     = range(10, 785)
# tdo_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 1
nu_s       = 1
derivative = true
nu         = 1
mu         = 1
filename   = directory_doe*"reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 785)
I_xt       = range(10, 785)
# tdo_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 1
nu_s       = 2
derivative = false
nu         = 2
mu         = 2
filename   = directory_doe*"reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 785)
I_xt       = range(10, 785)
# tdo_reparameterized_least_squares(filename, I_tauit, I_xt)

m        = 1
i_x      = 2
nu_s     = Inf
mu       = Inf
filename = directory_doe*"reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 785)
I_xt     = range(10, 785)
# tdo_reparameterized_least_squares(filename, I_tauit, I_xt)

m        = 1
nu_s     = 0
filename = directory_doe*"reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 785)
I_xt     = range(10, 785)
# tdo_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 2
nu_s       = 1
derivative = true
nu         = 1
mu         = 1
filename   = directory_doe*"reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 785)
I_xt       = range(10, 785)
# tdo_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 1
nu_s       = 2
derivative = false
nu         = 2
mu         = 2
filename   = directory_doe*"reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 785)
I_xt       = range(10, 785)
# tdo_reparameterized_least_squares(filename, I_tauit, I_xt)

function tdo_reparameterized_prediction_3D(t_f::Real, C_min::Real, C_max::Real, L::Real, nu_s::Real, nu_p::Real, normalize::Bool, standardize::Bool, n_C::Int, n_s::Int, filename::String)
    k  = MaternCovariance(nu_s, normalize=normalize, standardize=standardize) * MaternCovariance(nu_p, 2, normalize=normalize, standardize=standardize)
    Ch = collect(10 .^ range(log10(C_min), log10(C_max), length=n_C))
    sh = collect(range(0, 1, length=n_s))
    Ph = zeros(3, 0)

    for i in range(1, n_C)
        Ph = [Ph [sh'; repeat(Ch[i:i], 1, n_s); L*ones(1, n_s)]]
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
    for i in range(1, n_C)
        tauith[(i-1)*n_s+1] = 0
        tauith[i*n_s]       = t_f
    end

    display(heatmap(reshape(tauith, n_s, n_C)', xlabel=L"s", ylabel=L"C", colorbar=:none))

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

    display(heatmap(reshape(xth, n_s, n_C)', xlabel=L"s", ylabel=L"C", colorbar=:none))

    open(filename*"_prediction.csv", "w") do io
        write(io, "s,C,tauith,xth\n")
        DF.writedlm(io, [Ph[1,:] Ph[2,:] tauith xth], ',')
    end

    return nothing
end

m           = 1
t_f         = 1e-2
C_min       = 0.1e-6
C_max       = 10e-6
L           = 3e-3
i_x         = 1
nu_s        = Inf
nu_p        = Inf
normalize   = false
standardize = true
n_C         = 50
n_s         = 200
derivative  = false
nu          = 2
mu          = Inf
filename    = directory_doe*"reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# tdo_reparameterized_prediction_3D(t_f, C_min, C_max, L, nu_s, nu_p, normalize, standardize, n_C, n_s, filename)

m           = 2
i_x         = 2
nu_s        = 1
derivative  = true
nu          = 1
mu          = 1
filename    = directory_doe*"reparameterized_C_min=$(C_min)_C_max=$(C_max)_L_min=$(L_min)_L_max=$(L_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# tdo_reparameterized_prediction_3D(t_f, C_min, C_max, L, nu_s, nu_p, normalize, standardize, n_C, n_s, filename)

# computational cost
# t_conventional_3D          = [530, 567, 671, 268]
# t_reparameterized_3D_tauit = [572, 419, 504, 784]
# t_reparameterized_3D_xt    = [586, 472, 402, 801]
# t_reparameterized_3D       = [t_reparameterized_3D_tauit; t_reparameterized_3D_xt]
# @info "t_x = $(Statistics.mean(t_conventional_3D)) +- $(Statistics.std(t_conventional_3D))"
# @info "t_tauit = $(Statistics.mean(t_reparameterized_3D_tauit)) +- $(Statistics.std(t_reparameterized_3D_tauit))"
# @info "t_xt = $(Statistics.mean(t_reparameterized_3D_xt)) +- $(Statistics.std(t_reparameterized_3D_xt))"
# @info "t_xt(tauit) = $(Statistics.mean(t_reparameterized_3D)) +- $(Statistics.std(t_reparameterized_3D))"
