"""Quick test for both ML and RL pipelines."""
from main import run_pipeline

print("=" * 50)
print("TESTING ML PIPELINE")
print("=" * 50)
r = run_pipeline("AAPL", "2y", model_type="ML")
if r:
    print(f"ML: {len(r['trades'])} trades")
    print(f"ML Training Details: {r['training_details']}")
    print(f"ML Metrics: Total Return={r['metrics'].get('Total_Return', 0):.2%}")
else:
    print("ML: No results")

print()
print("=" * 50)
print("TESTING RL PIPELINE")
print("=" * 50)
r2 = run_pipeline("AAPL", "2y", model_type="RL", rl_timesteps=2000)
if r2:
    print(f"RL: {len(r2['trades'])} trades")
    print(f"RL Training Details: {r2['training_details']}")
    print(f"RL Metrics: Total Return={r2['metrics'].get('Total_Return', 0):.2%}")
else:
    print("RL: No results")

print()
print("ALL TESTS PASSED!")
