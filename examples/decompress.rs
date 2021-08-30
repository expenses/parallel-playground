use parallel_playground::*;

fn main() {
    let input =  [5, 4, 0, 3, 5, 5, 5, 5, 5, 5];

    let scanned_input = exclusive_scan(&input);

    let output_size = sum(&input);

    let mut output = zeroed_array_of_size(output_size);

    println!("{:?}", scanned_input);

    scatter_with_value(&mut output, &scanned_input, 1);

    println!("{:?}", output);

    let x = inclusive_scan(&output);

    println!("{:?}", x);

    let mapped = x.iter().map(|&v| v % 2).collect::<Vec<_>>();

    println!("{:?}", mapped);
}
