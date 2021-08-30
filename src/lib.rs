pub fn sum(input: &[u32]) -> u32 {
    input.iter().sum()
}

pub fn inclusive_scan(input: &[u32]) -> Vec<u32> {
    input.iter().scan(0, |state, i| {
        *state += i;

        Some(*state)
    }).collect()
}

pub fn exclusive_scan(input: &[u32]) -> Vec<u32> {
    input.iter().scan(0, |state, i| {
        let prev_state = *state;

        *state += i;

        Some(prev_state)
    }).collect()
}

pub fn zeroed_array_of_size(size: u32) -> Vec<u32> {
    vec![0; size as usize]
}

pub fn scatter_if_non_zero(output: &mut [u32], indices: &[u32], input: &[u32]) {
    for i in (0 .. input.len()).rev() {
        let value = input[i];
        if value > 0 {
            output[indices[i] as usize] = input[i]
        }
    }
}

pub fn scatter_with_value(output: &mut [u32], indices: &[u32], value: u32) {
    for i in (0 .. indices.len()).rev() {
        output[indices[i] as usize] = value;
    }
}

pub fn ascending_values_of_len(len: u32) -> Vec<u32> {
    (0 .. len).map(|_| 1).collect()
}

