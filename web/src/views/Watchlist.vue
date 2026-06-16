<template>
  <div class="page-head">
    <div>
      <h1 class="page-title">Stocks &amp; Watchlist</h1>
      <p class="page-sub">Filter stocks and build your personal watchlist</p>
    </div>
  </div>

  <!-- Filter controls -->
  <div class="controls" style="margin-bottom:1.5rem; flex-wrap:wrap">
    <input v-model="search" placeholder="Search symbol or company..." style="flex:1; min-width:220px" />
    <select v-model="sector">
      <option value="">All Sectors</option>
      <option v-for="s in sectors" :key="s" :value="s">{{ s }}</option>
    </select>
    <label class="page-sub" style="display:flex; align-items:center; gap:.4rem; cursor:pointer">
      <input type="checkbox" v-model="watchOnly" style="width:auto" /> Watchlist only
    </label>
  </div>

  <div v-if="!filtered.length" class="page-sub">No stocks match your filters.</div>

  <div class="stocks">
    <div class="stock" v-for="s in filtered" :key="s.symbol">
      <div class="top">
        <div>
          <div class="sym">{{ s.symbol }}</div>
          <div class="nm">{{ s.name }}</div>
        </div>
        <button class="star" :class="{ on: s.tracked }" @click="toggle(s)">{{ s.tracked ? '★' : '☆' }}</button>
      </div>
      <div style="margin-top:1rem; display:flex; justify-content:space-between; align-items:center;">
        <span class="nm">{{ s.sector }}</span>
        <div style="text-align:right">
          <div style="font-weight:600">PKR {{ s.price }}</div>
          <div :class="s.change >= 0 ? 'green' : 'red'" style="font-size:.8rem">
            {{ s.change >= 0 ? '+' : '' }}{{ s.change }}%
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import api from '../api'

const stocks = ref([])
const search = ref('')
const sector = ref('')
const watchOnly = ref(false)

// Build the sector dropdown from whatever stocks are loaded.
const sectors = computed(() => [...new Set(stocks.value.map((s) => s.sector))].sort())

// Apply search + sector + watchlist-only filters.
const filtered = computed(() =>
  stocks.value.filter((s) => {
    const q = search.value.toLowerCase()
    const matchesSearch = !q || s.symbol.toLowerCase().includes(q) || s.name.toLowerCase().includes(q)
    const matchesSector = !sector.value || s.sector === sector.value
    const matchesWatch = !watchOnly.value || s.tracked
    return matchesSearch && matchesSector && matchesWatch
  })
)

async function loadStocks() {
  const { data } = await api.get('/api/stocks')
  stocks.value = data.stocks
}

async function toggle(s) {
  if (s.tracked) await api.delete(`/api/watchlist/${s.symbol}`)
  else await api.post(`/api/watchlist/${s.symbol}`)
  s.tracked = !s.tracked
}

onMounted(loadStocks)
</script>
