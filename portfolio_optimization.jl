############################################################
# Portfolio Optimization with Machine Learning in Julia
#
# Metode:
# 1. Mean-Variance Optimization (JuMP + OSQP)
# 2. Reinforcement Learning (Q-Learning, kasus sederhana 2 aset)
# 3. Asset Return Prediction menggunakan MLJ
#
# Teknologi: JuMP.jl, DataFrames.jl, MLJ.jl, Plots.jl
############################################################

# Bagian 0: Import Package yang Diperlukan
using JuMP
using OSQP
using LinearAlgebra
using DataFrames
using Random
using Statistics
using Plots

# Untuk bagian MLJ, pastikan paket berikut sudah diinstal:
using MLJ
using MLJLinearModels

############################################################
# Bagian 1: Mean-Variance Optimization dengan JuMP
############################################################

println("== Mean-Variance Optimization ==")

# Misalkan kita memiliki 4 aset
assets = ["Asset A", "Asset B", "Asset C", "Asset D"]
n_assets = length(assets)

# Data sintetis: expected returns dan covariance matrix
expected_returns = [0.10, 0.12, 0.14, 0.09]  # misal: 10%, 12%, 14%, 9%
Sigma = [0.005   0.001   0.002   0.001;
         0.001   0.004   0.0015  0.0005;
         0.002   0.0015  0.006   0.001;
         0.001   0.0005  0.001   0.003]

# Target return portofolio
target_return = 0.11

# Buat model optimasi dengan OSQP (solver untuk QP)
model = Model(OSQP.Optimizer)
@variable(model, w[1:n_assets] >= 0)
@constraint(model, sum(w) == 1)
@constraint(model, dot(expected_returns, w) >= target_return)
@objective(model, Min, dot(w, Sigma * w))
optimize!(model)

w_opt = value.(w)
df_opt = DataFrame(Asset = assets, Weight = w_opt)
println("Optimal Portfolio Weights (Mean-Variance):")
println(df_opt)

# Visualisasikan bobot optimal
bar(assets, w_opt, title="Optimal Portfolio Weights (Mean-Variance)", ylabel="Weight", legend=false)
savefig("optimal_weights.png")  # simpan plot ke file

############################################################
# Bagian 2: Reinforcement Learning untuk Seleksi Portofolio (Kasus 2 Aset)
############################################################

println("\n== Reinforcement Learning for Portfolio Selection ==")
# Untuk kesederhanaan, kita anggap hanya ada 2 aset.
# Setiap episode (misal: 1 hari) agen memilih satu aset untuk diinvestasikan.
# Reward adalah return harian aset tersebut.

# Parameter simulasi
n_episodes = 1000  # jumlah episode (hari)
n_assets_rl = 2    # 2 aset
true_means = [0.08, 0.10]  # rata-rata return sebenarnya untuk masing-masing aset
true_std   = [0.02, 0.025] # standar deviasi return

# Parameter Q-Learning
alpha = 0.1    # learning rate
gamma = 0.99   # discount factor (tidak terlalu berpengaruh pada bandit stateless)
epsilon = 0.1  # tingkat eksplorasi

# Inisialisasi nilai Q untuk masing-masing aksi (aset)
Q = zeros(n_assets_rl)
reward_history_rl = zeros(n_episodes)

# Q-Learning Loop (masalah bandit stateless)
for episode in 1:n_episodes
    # Pilih aksi dengan strategi epsilon-greedy
    if rand() < epsilon
        action = rand(1:n_assets_rl)
    else
        action = argmax(Q)
    end

    # Simulasi return aset (dari distribusi normal)
    reward = rand(Normal(true_means[action], true_std[action]))
    reward_history_rl[episode] = reward

    # Update Q-value (tanpa state transition karena stateless)
    Q[action] = Q[action] + alpha * (reward - Q[action])
end

println("Learned Q-values untuk masing-masing aset:")
for i in 1:n_assets_rl
    println("Asset ", i, ": Q-value = ", round(Q[i], digits=4))
end

best_asset = argmax(Q)
println("Aset terbaik menurut RL: Asset ", best_asset)

# Visualisasikan reward history selama pelatihan RL
plot(1:n_episodes, reward_history_rl, title="RL Reward History", xlabel="Episode", ylabel="Reward", legend=false)
savefig("rl_reward_history.png")

############################################################
# Bagian 3: Prediksi Return Aset menggunakan MLJ
############################################################

println("\n== Asset Return Prediction using MLJ ==")
# Simulasikan data untuk prediksi return aset.
# Misalkan kita punya fitur: past_return dan market_return.
n_samples = 200
X = DataFrame(past_return = randn(n_samples), market_return = randn(n_samples))
# Hubungan sebenarnya: return = 0.05 + 0.3*past_return + 0.5*market_return + noise
y = 0.05 .+ 0.3 .* X.past_return .+ 0.5 .* X.market_return .+ 0.1*randn(n_samples)

# Definisikan model regresi linear dari MLJLinearModels
model_lr = LinearRegressor()
mach = machine(model_lr, X, y)
fit!(mach)

# Prediksi untuk data baru
X_new = DataFrame(past_return = [0.1, -0.2, 0.05], market_return = [0.2, 0.0, -0.1])
y_pred = predict(mach, X_new)
println("Prediksi return untuk data baru:")
println(y_pred)

############################################################
# Selesai
############################################################

println("\nPortfolio Optimization with Machine Learning completed.")
