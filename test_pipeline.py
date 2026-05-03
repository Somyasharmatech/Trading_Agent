"""Test model persistence: train → save → load → predict."""
import os
from main import run_pipeline, train_rl_only
from rl_agent import is_model_saved, load_rl_model

# Clean up any old model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_model.zip")
if os.path.exists(model_path):
    os.remove(model_path)
    print("Cleaned old model.")

print("=" * 60)
print("TEST 1: Train RL model and save")
print("=" * 60)
stats = train_rl_only("AAPL", period="2y", rl_timesteps=5000)
print(f"Training done. Episodes: {stats['total_episodes']}, Avg Reward: {stats['avg_reward']:.4f}")
print(f"Model saved: {is_model_saved()}")
assert is_model_saved(), "Model should be saved!"

print()
print("=" * 60)
print("TEST 2: Load saved model")
print("=" * 60)
model = load_rl_model()
assert model is not None, "Model should load!"
print("Model loaded successfully.")

print()
print("=" * 60)
print("TEST 3: Run simulation with saved model (FAST)")
print("=" * 60)
import time
start = time.time()
r = run_pipeline("AAPL", "2y", model_type="RL", rl_timesteps=5000, use_saved_model=True)
elapsed = time.time() - start
print(f"Fast simulation: {len(r['trades'])} trades in {elapsed:.1f}s")
print(f"Mode: {r['training_details'].get('mode', 'N/A')}")

print()
print("=" * 60)
print("TEST 4: ML pipeline still works")
print("=" * 60)
r2 = run_pipeline("AAPL", "2y", model_type="ML")
print(f"ML: {len(r2['trades'])} trades, Win Rate: {r2['metrics'].get('Win_Rate', 0):.2%}")

print()
print("ALL TESTS PASSED!")
