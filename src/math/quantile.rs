use itertools::Itertools;

/// Return quantile values of a given slice.
pub fn quantiles<T>(values: &[T], quantiles: &[f64]) -> Vec<T>
where
    T: Clone + PartialOrd,
{
    let values_sorted = values
        .iter()
        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
        .cloned()
        .collect::<Vec<T>>();

    let mut results = Vec::with_capacity(quantiles.len());

    for quantile in quantiles {
        results.push(values_sorted[(values.len() as f64 * quantile) as usize].clone());
    }

    results
}

/// Return the larget `n` values of a slice.
pub fn largest_n<T>(values: &[T], n: usize) -> Vec<T>
where
    T: Clone + PartialOrd,
{
    let valus_sorted = values
        .iter()
        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
        .cloned()
        .collect::<Vec<T>>();

    Vec::from(&valus_sorted[(values.len() - n)..])
}
