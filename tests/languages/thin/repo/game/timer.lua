-- Countdown timers for the game loop.
local M = {}

local TICK = 1 / 60

local function clamp(x, lo, hi)
  if x < lo then return lo end
  if x > hi then return hi end
  return x
end

function M.new(seconds)
  return { left = seconds }
end

function M.step(t)
  t.left = clamp(t.left - TICK, 0, math.huge)
  return t.left == 0
end

return M
