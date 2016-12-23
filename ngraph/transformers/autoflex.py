# algorithm copied from neon flexsim
  
def init_scale_algorithm(maxabs, scale, init_count, high_bit):

    initialized = False
    high = (1 << high_bit) - 1  # 32768 - 1
    low = 1 << high_bit - 1  # 16384

    if maxabs >= high:
        # we saturated because our scale was too small.Because saturation
        # destroys the information about how small, just drop back half of
        # our bit width
        scale *= 2.0 ** (high_bit >> 1)

    elif maxabs < low:
        # we underutilized (or underflowed) because our scale was too big
        scale *= 2.0 ** (len(bin(maxabs)) - len(bin(low)))
        # If maxabs is greater than zero we should be able to jump straight
        # to full bit utilization but wait till we collect enough bits for
        # this to be a reliable jump. Note that this logic is tuned to
        # prevent ping ponging back and forth.
        if maxabs >= (1 << max((high_bit >> 1)-2, 0)):  # for flex16 works out to 32, or 5 bits
            count = 0
            initialized = True

    # we're in the zone
    else:
        count = 0
        initialized = True

    count = init_count + 1

    # probably all zero input, just set things up so it can adjust when input does come in.
    if count > 10:
        scale = 1.0
        count = 0  # reset back in case we repeat the init procedure
        initialized = True

    # RP TODO: reinstate setting children (input scales) here again?

    return scale, count, initialized
