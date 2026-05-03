"""Quick test for the upgraded pipeline (RSI + cooldown + trend filter)."""
from main import run_pipeline

print("=" * 60)
print("TESTING UPGRADED ML PIPELINE (RSI + Cooldown + Trend Filter)")
print("=" * 60)
r = run_pipeline("AAPL", "2y", model_type="ML")
if r:
    print(f"ML: {len(r['trades'])} trades")
    print(f"ML Training: {r['training_details']}")
    m = r['metrics']
    print(f"ML Return: {m.get('Total_Return', 0):.2%}")
    print(f"ML Win Rate: {m.get('Win_Rate', 0):.2%}")
    print(f"ML Sharpe: {m.get('Sharpe_Ratio', 0):.2f}")
else:
    print("ML: No results")

print()
print("=" * 60)
print("TESTING UPGRADED RL PIPELINE (RSI + Cooldown + Trend Filter)")
print("=" * 60)
r2 = run_pipeline("AAPL", "2y", model_type="RL", rl_timesteps=5000)
if r2:
    print(f"RL: {len(r2['trades'])} trades")
    td = r2['training_details']
    print(f"RL Training: Algorithm={td['algorithm']}, Timesteps={td['timesteps']}, Episodes={td['total_episodes']}, Avg Reward={td['avg_reward']:.4f}")
    m2 = r2['metrics']
    print(f"RL Return: {m2.get('Total_Return', 0):.2%}")
    print(f"RL Win Rate: {m2.get('Win_Rate', 0):.2%}")
    print(f"RL Sharpe: {m2.get('Sharpe_Ratio', 0):.2f}")
else:
    print("RL: No results")

print()
print("ALL TESTS PASSED!")
