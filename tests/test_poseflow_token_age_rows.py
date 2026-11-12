import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.rows_token_age import (
    PAD, PERSON, PREDICT, build_training_rows, token_age_sigmas,
)


def seeded_generator(seed=42):
    return torch.Generator().manual_seed(seed)


def test_age_schedule_uses_noise_coefficient_rather_than_variance():
    actual = token_age_sigmas(torch.arange(5))
    torch.testing.assert_close(actual, torch.tensor([1., .1, .05, .025, .0125]))
    with pytest.raises(ValueError, match='age'):
        token_age_sigmas(torch.tensor([-1]))


def test_one_permutation_preserves_coordinates_missing_joint_masks_and_identities():
    clean = torch.arange(2*5*32, dtype=torch.float32).reshape(2, 5, 32)/100
    counts = torch.tensor([5, 2])
    masks = torch.ones_like(clean, dtype=torch.bool)
    masks[0, 1, :6] = False
    identities = torch.tensor([[10, 11, 12, 13, 14], [20, 21, -1, -1, -1]])
    original = clean.clone()
    result = build_training_rows(clean, counts, coordinate_mask=masks,
                                 target_indices=identities, generator=seeded_generator())
    sources = result['sources']
    expected_generator = seeded_generator()
    for image, count in enumerate(counts.tolist()):
        order = torch.randperm(count, generator=expected_generator)
        assert sources['source_indices'][image, :count].tolist() == order.tolist()
        torch.testing.assert_close(sources['coordinates'][image, :count], clean[image, order])
        assert torch.equal(sources['coordinate_mask'][image, :count], masks[image, order])
        assert torch.equal(sources['target_indices'][image, :count], identities[image, order])
        assert sources['source_indices'][image, count:].eq(-1).all()
        assert sources['target_indices'][image, count:].eq(-1).all()
        assert not sources['coordinate_mask'][image, count:].any()
        assert sources['coordinates'][image, count:].eq(0).all()
    assert not torch.equal(sources['source_indices'][0], torch.arange(5))
    assert torch.equal(clean, original)


@pytest.mark.parametrize('counts', [[0], [1], [3, 1, 0], [49]])
def test_growth_all_active_supervision_and_full_eos_refinement_tail(counts):
    result = build_training_rows(torch.zeros(len(counts), 50, 32), torch.tensor(counts),
                                 generator=seeded_generator())
    assert result['noisy_coordinates'].shape == (len(counts), 50, 50, 32)
    for image, count in enumerate(counts):
        assert result['birth_mask'][image].sum() == count
        assert result['eos_targets'][image].sum() == 50-count
        for row in range(50):
            prefix = min(row, count)
            types = result['token_types'][image, row]
            assert types[:prefix].eq(PERSON).all()
            assert types[prefix] == PREDICT
            assert types[prefix+1:].eq(PAD).all()
            assert result['predict_indices'][image, row] == prefix
            assert result['refinement_mask'][image, row].sum() == prefix
            assert result['supervised'][image, row].sum() == min(row+1, count)
            assert bool(result['eos_targets'][image, row]) == (row >= count)
        for rank in range(count):
            assert result['supervised'][image, :, rank].sum() == 50-rank
    assert torch.equal(result['padding_mask'], result['token_types'] == PAD)
    assert torch.equal(result['supervised'], result['birth_mask'] | result['refinement_mask'])


def test_corruption_oracle_per_token_conditioning_and_final_clean_endpoint():
    result = build_training_rows(torch.ones(1, 5, 32), torch.tensor([4]),
                                 generator=seeded_generator())
    torch.testing.assert_close(result['noise_sigmas'][0, 3, :4],
                               torch.tensor([.025, .05, .1, 1.]))
    clean = torch.nn.functional.pad(result['sources']['coordinates'], (0, 0, 0, 45))
    sigma = result['noise_sigmas'][..., None]
    expected = (1-sigma.square()).sqrt()*clean[:, None]+sigma*result['noise']
    expected = expected.masked_fill(result['padding_mask'][..., None], 0)
    torch.testing.assert_close(result['noisy_coordinates'], expected, atol=0, rtol=0)
    torch.testing.assert_close(result['noisy_coordinates'][0, -1, :4],
                               result['sources']['coordinates'][0, :4], atol=0, rtol=0)
    assert result['noise_sigmas'][0, -1, :4].eq(0).all()


def test_birth_and_eos_queries_both_contain_pure_noise_and_padding_contains_nothing():
    result = build_training_rows(torch.full((1, 4, 32), 7.), torch.tensor([2]),
                                 generator=seeded_generator())
    query = result['token_types'] == PREDICT
    assert result['noise_sigmas'][query].eq(1).all()
    torch.testing.assert_close(result['noisy_coordinates'][query], result['noise'][query],
                               atol=0, rtol=0)
    padded = result['padding_mask']
    assert result['noise_sigmas'][padded].eq(0).all()
    assert result['noise'][padded].eq(0).all()
    assert result['noisy_coordinates'][padded].eq(0).all()
    assert not result['supervised'][query & result['eos_targets'][..., None]].any()
    assert not torch.equal(result['noise'][0, 1, 0], result['noise'][0, 2, 0])
    assert not torch.equal(result['noise'][0, 2, 0], result['noise'][0, 2, 1])


def test_future_sources_and_padded_values_cannot_leak_into_earlier_rows():
    clean = torch.rand(1, 6, 32)
    result = build_training_rows(clean, torch.tensor([4]), generator=seeded_generator())
    changed = clean.clone()
    future_source = result['sources']['source_indices'][0, 2]
    changed[0, future_source] += 1000
    changed[0, 4:] = float('nan')
    other = build_training_rows(changed, torch.tensor([4]), generator=seeded_generator())
    torch.testing.assert_close(result['noisy_coordinates'][:, :3],
                               other['noisy_coordinates'][:, :3], atol=0, rtol=0)
    assert not torch.equal(result['noisy_coordinates'][:, 3], other['noisy_coordinates'][:, 3])
    assert torch.isfinite(other['noisy_coordinates']).all()


def test_explicit_generator_repeats_rows_without_advancing_global_rng():
    clean = torch.ones(2, 5, 32)
    state = torch.get_rng_state().clone()
    first = build_training_rows(clean, torch.tensor([5, 3]), generator=seeded_generator())
    second = build_training_rows(clean, torch.tensor([5, 3]), generator=seeded_generator())
    assert torch.equal(state, torch.get_rng_state())
    for key in ('noisy_coordinates', 'noise_sigmas', 'noise'):
        assert torch.equal(first[key], second[key])
    third = build_training_rows(clean, torch.tensor([5, 3]), generator=seeded_generator(43))
    assert not torch.equal(first['noise'], third['noise'])


def test_empty_source_storage_and_missing_joint_masks_remain_valid():
    result = build_training_rows(torch.empty(1, 0, 32), torch.tensor([0]),
                                 generator=seeded_generator())
    assert result['eos_targets'].all()
    assert not result['supervised'].any()
    assert result['sources']['coordinates'].shape == (1, 0, 32)
    assert not result['padding_mask'].all(-1).any()


@pytest.mark.parametrize('counts, row_count, message', [
    (torch.tensor([50]), 50, 'capacity'),
    (torch.tensor([4]), 4, 'capacity'),
    (torch.tensor([-1]), 50, 'counts'),
    (torch.tensor([1.]), 50, 'counts'),
    (torch.tensor([0]), 1, 'row'),
])
def test_invalid_counts_or_row_budget_rejected_before_rng_is_used(counts, row_count, message):
    generator = seeded_generator()
    state = generator.get_state().clone()
    with pytest.raises(ValueError, match=message):
        build_training_rows(torch.zeros(1, 50, 32), counts, row_count=row_count,
                            generator=generator)
    assert torch.equal(state, generator.get_state())
